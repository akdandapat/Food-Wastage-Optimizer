type KpiCardProps = {
  label: string;
  value: string;
  caption: string;
  tone?: "warm" | "cool" | "earth";
};

export default function KpiCard({
  label,
  value,
  caption,
  tone = "warm"
}: KpiCardProps) {
  return (
    <article className={`kpi-card tone-${tone}`}>
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      <p className="kpi-caption">{caption}</p>
    </article>
  );
}

