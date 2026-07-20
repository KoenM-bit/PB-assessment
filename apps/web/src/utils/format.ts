export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("nl-NL", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("nl-NL");
}

export function formatPercent(value: number | null): string {
  if (value === null) return "—";
  return `${value.toFixed(1)}%`;
}

export function formatDurationMs(ms: number | null): string {
  if (ms === null || ms <= 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** Compact € in thousands, e.g. 598k or 22.9k */
export function formatEuroK(value: number, signed = false): string {
  const k = value / 1000;
  const absStr =
    Math.abs(k) >= 100 ? String(Math.round(Math.abs(k))) : Math.abs(k).toFixed(1);
  if (signed) {
    if (value > 0) return `+${absStr}k`;
    if (value < 0) return `-${absStr}k`;
    return "0k";
  }
  return `${absStr}k`;
}

export function improvementEur(baselineErr: number, modelErr: number): number {
  return baselineErr - modelErr;
}
