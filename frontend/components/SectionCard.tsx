import { ReactNode } from "react";

type SectionCardProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export default function SectionCard({
  title,
  subtitle,
  children
}: SectionCardProps) {
  return (
    <section className="section-card">
      <div className="section-heading">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

