import Head from "next/head";
import { ReactNode } from "react";

type LayoutProps = {
  children: ReactNode;
};

export default function Layout({ children }: LayoutProps) {
  return (
    <>
      <Head>
        <title>Kitchen Demand Command Center</title>
        <meta
          name="description"
          content="Kitchen-level demand forecasting, optimization, and MLOps dashboard for Kolkata hostel mess operations."
        />
      </Head>
      <div className="page-shell">
        <header className="hero-panel">
          <div>
            <p className="eyebrow">Spatiotemporal Demand Forecasting</p>
            <h1>Kolkata Hostel Kitchen Command Center</h1>
            <p className="hero-copy">
              A multi-kitchen forecasting, optimization, and feedback system
              for university hostel mess operations across Kolkata.
            </p>
          </div>
          <div className="hero-stat-block">
            <span>Pipeline</span>
            <strong>Raw data -&gt; Models -&gt; Optimization -&gt; Feedback loop</strong>
          </div>
        </header>
        <main className="content-grid">{children}</main>
      </div>
    </>
  );
}
