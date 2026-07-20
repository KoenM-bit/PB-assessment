import type { LiveLabelledSale } from "../types";

export interface LiveSalesHeadToHead {
  sample_size: number;
  model_better_pct: number;
  baseline_better_pct: number;
  avg_win_when_model_better_eur: number | null;
  avg_loss_when_baseline_better_eur: number | null;
  model_wins: number;
  baseline_wins: number;
  ties: number;
}

export function computeLiveSalesHeadToHead(items: LiveLabelledSale[]): LiveSalesHeadToHead {
  if (items.length === 0) {
    return {
      sample_size: 0,
      model_better_pct: 0,
      baseline_better_pct: 0,
      avg_win_when_model_better_eur: null,
      avg_loss_when_baseline_better_eur: null,
      model_wins: 0,
      baseline_wins: 0,
      ties: 0,
    };
  }

  const wins: number[] = [];
  const losses: number[] = [];
  let ties = 0;

  for (const row of items) {
    const delta = row.baseline_abs_error - row.model_abs_error;
    if (delta > 0) {
      wins.push(delta);
    } else if (delta < 0) {
      losses.push(-delta);
    } else {
      ties += 1;
    }
  }

  const decided = items.length - ties;
  const model_wins = wins.length;
  const baseline_wins = losses.length;
  const model_better_pct = decided > 0 ? (model_wins / decided) * 100 : 0;
  const baseline_better_pct = decided > 0 ? (baseline_wins / decided) * 100 : 0;

  const avg = (values: number[]) =>
    values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;

  return {
    sample_size: items.length,
    model_better_pct: Math.round(model_better_pct * 10) / 10,
    baseline_better_pct: Math.round(baseline_better_pct * 10) / 10,
    avg_win_when_model_better_eur: avg(wins),
    avg_loss_when_baseline_better_eur: avg(losses),
    model_wins,
    baseline_wins,
    ties,
  };
}
