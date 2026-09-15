/** Small helpers the SVG charts share: round axis ticks, label widths, clipping. */

/** Round tick values for the current scale: about five of them, at 1 / 2 / 2.5 / 5 × a power of ten. */
export function axisTicks(max: number): number[] {
  const raw = max / 5;
  const magnitude = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1))));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((candidate) => candidate >= raw) ?? 10 * magnitude;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-6; v += step) out.push(Math.round(v * 10) / 10);
  return out;
}

/** Inter at these sizes runs about 0.42 em per character. */
export function textWidth(text: string, size: number): number {
  return text.length * size * 0.42;
}

/** Keep a label inside its column; an over-long one ends in an ellipsis. */
export function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
}
