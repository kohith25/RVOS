import os
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rl_grpo.arguments import RLGRPOArguments
from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.rewards import RewardWeights, compute_composite_reward
from rl_grpo.trajectory import MultiTurnTrajectoryGenerator
from rl_grpo.utils import parse_gt_bbox


def _extract_text_from_sample(tokenizer, batch_input_ids, batch_labels, idx: int) -> Dict[str, str]:
    instruction_ids = batch_input_ids[idx].tolist()
    label_ids = batch_labels[idx].tolist()

    instruction_text = tokenizer.decode(instruction_ids, skip_special_tokens=False)

    out_ids = [t for t in label_ids if t != -100]
    output_text = tokenizer.decode(out_ids, skip_special_tokens=False) if out_ids else ""

    # Parse expression for state; fallback to full instruction if regex fails.
    expression = instruction_text
    marker = "Given the referential expression"
    if marker in instruction_text:
        expression = instruction_text.split(marker, 1)[-1]

    return {
        "instruction": instruction_text,
        "expression": expression,
        "output": output_text,
    }


def _build_action_supervision(traj) -> str:
    lines = []
    for t in traj.turns:
        lines.append(f"{{\"action\":\"{t.action}\",\"observation\":\"{t.observation}\"}}")
    lines.append(traj.final_output)
    return "\n".join(lines)


def run_grpo_training(
    model,
    tokenizer,
    train_dataset,
    data_collator,
    training_args,
    output_dir: str,
):
    device = next(model.parameters()).device

    rl_args = RLGRPOArguments(
        enabled=True,
        max_turn=training_args.rl_max_turn,
        num_generations=training_args.rl_num_generations,
        max_steps=training_args.rl_max_steps,
        learning_rate=training_args.rl_learning_rate,
        weight_decay=training_args.rl_weight_decay,
        grad_clip=training_args.rl_grad_clip,
        reward_wf=training_args.rl_reward_wf,
        reward_wt=training_args.rl_reward_wt,
        reward_wm=training_args.rl_reward_wm,
        reward_we=training_args.rl_reward_we,
        reward_wp=training_args.rl_reward_wp,
        reward_wc=training_args.rl_reward_wc,
        lambda_seg=training_args.rl_lambda_seg,
        lambda_reason=training_args.rl_lambda_reason,
        temperature=training_args.rl_temperature,
        top_p=training_args.rl_top_p,
        max_action_tokens=training_args.rl_max_action_tokens,
        logging_steps=training_args.rl_logging_steps,
        save_steps=training_args.rl_save_steps,
    )

    adapter = TGSModelAdapter(model=model, tokenizer=tokenizer, device=device)
    trajectory_generator = MultiTurnTrajectoryGenerator(
        adapter=adapter,
        max_turn=rl_args.max_turn,
        temperature=rl_args.temperature,
        top_p=rl_args.top_p,
        max_action_tokens=rl_args.max_action_tokens,
    )

    reward_weights = RewardWeights(
        wf=rl_args.reward_wf,
        wt=rl_args.reward_wt,
        wm=rl_args.reward_wm,
        we=rl_args.reward_we,
        wp=rl_args.reward_wp,
        wc=rl_args.reward_wc,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=rl_args.learning_rate,
        weight_decay=rl_args.weight_decay,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=training_args.dataloader_num_workers,
        drop_last=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "rl_grpo_config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(asdict(rl_args), f, indent=2)

    model.train()
    global_step = 0

    while global_step < rl_args.max_steps:
        for batch in dataloader:
            if global_step >= rl_args.max_steps:
                break

            batch_input_ids = batch["batch_input_ids"]
            batch_labels = batch["batch_labels"]
            bsz = len(batch_input_ids)

            total_loss = torch.tensor(0.0, device=device)
            total_reward = 0.0

            for i in range(bsz):
                sample = _extract_text_from_sample(tokenizer, batch_input_ids, batch_labels, i)
                gt_box = parse_gt_bbox(sample["output"])

                trajectories = []
                rewards = []

                for _ in range(rl_args.num_generations):
                    traj = trajectory_generator.rollout(
                        instruction=sample["instruction"],
                        expression=sample["expression"],
                        gt_box=gt_box,
                    )
                    reward_dict = compute_composite_reward(traj, gt_box, reward_weights)
                    trajectories.append(traj)
                    rewards.append(reward_dict["R_total"])

                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                adv = rewards_t - rewards_t.mean()
                std = rewards_t.std(unbiased=False)
                if std.item() > 1e-6:
                    adv = adv / (std + 1e-6)

                sample_loss = torch.tensor(0.0, device=device)

                for j, traj in enumerate(trajectories):
                    prompt = sample["instruction"] + "\n<rl_trace>\n"
                    completion = _build_action_supervision(traj)
                    logp_sum, token_count = adapter.sequence_logprob_sum(prompt, completion)
                    if token_count == 0:
                        continue
                    sample_loss = sample_loss - adv[j].detach() * (logp_sum / token_count)

                sample_loss = sample_loss / max(1, rl_args.num_generations)
                total_loss = total_loss + sample_loss
                total_reward += float(rewards_t.mean().item())

            total_loss = total_loss / max(1, bsz)
            total_loss.backward()

            if rl_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rl_args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % rl_args.logging_steps == 0:
                avg_r = total_reward / max(1, bsz)
                print(f"[RL-GRPO] step={global_step} loss={total_loss.item():.6f} reward={avg_r:.4f}")

            if global_step % rl_args.save_steps == 0:
                ckpt_dir = os.path.join(output_dir, f"rl-checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    final_dir = os.path.join(output_dir, "rl-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("RL-GRPO training finished.")
