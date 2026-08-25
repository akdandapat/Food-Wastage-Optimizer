import Head from "next/head";
import { ReactNode } from "react";

type LayoutProps = {
  children: ReactNode;
};

export default function Layout({ children }: LayoutProps) {
  return (
    <>
      <Head>
        <title>JU Mess Demand Dashboard</title>
        <meta
          name="description"
          content="Demand forecasting, optimization, and MLOps dashboard for Jadavpur University hostel messes."
        />
      </Head>
      <div className="page-shell">
        <header className="hero-panel">
          <div>
            <p className="eyebrow">JU Hostel Mess Forecasting</p>
            <h1>JU Mess Demand Dashboard</h1>
            <p className="hero-copy">
              Demand forecasting, Newsvendor optimization, and feedback loop
              for 3 Jadavpur University hostel messes.
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
