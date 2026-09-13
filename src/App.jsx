import React, { useState } from "react";

import {
  LayoutDashboard,
  PlusSquare,
  List,
  BarChart3,
  Settings,
  CircleHelp,
  Bell,
  Search,
  ChevronDown,
  ChevronRight,
  Folder,
  Zap,
  Database,
  FileText,
  BrainCircuit,
  Users,
  Star,
  Clock3,
  Crown,
  Sparkles,
  Check,
  Info,
  Box,
  SlidersHorizontal,
} from "lucide-react";

const initialForm = {
  description: "",
  affected: "1",
  downtime: "No downtime",
  errorRate: "0% (no errors)",
  productArea: "",
  customerTier: "Standard",
  serviceDisruption: false,
  securityRelated: false,
  revenueImpact: false,
  vipCustomer: false,
};

function SidebarItem({ icon: Icon, children, active, badge }) {
  return (
    <button className={`sidebar-item ${active ? "active" : ""}`}>
      <Icon size={21} strokeWidth={1.8} />
      <span>{children}</span>

      {badge !== undefined && (
        <span className="sidebar-badge">{badge}</span>
      )}
    </button>
  );
}

function GlassIcon({ icon: Icon, variant = "purple" }) {
  return (
    <div className={`glass-icon glass-icon-${variant}`}>
      <div className="glass-icon-highlight" />
      <Icon size={27} strokeWidth={1.8} />
    </div>
  );
}

function MetricCard({ icon, title, value, variant = "purple" }) {
  return (
    <section className="metric-card glass-panel">
      <GlassIcon icon={icon} variant={variant} />

      <div className="metric-content">
        <span className="metric-title">{title}</span>
        <strong>{value}</strong>

        <div className="metric-progress">
          <span className={`metric-fill ${variant}`} />
        </div>
      </div>
    </section>
  );
}

function SelectBox({
  label,
  icon: Icon,
  name,
  value,
  onChange,
  options,
  required,
}) {
  return (
    <label className="form-field">
      <span className="field-label">
        {Icon && <Icon size={18} strokeWidth={1.7} />}
        {label}
        {required && <b>*</b>}
      </span>

      <div className="select-wrap">
        <select name={name} value={value} onChange={onChange}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>

        <ChevronDown size={15} />
      </div>
    </label>
  );
}

function Toggle({ label, name, checked, onChange }) {
  return (
    <label className="toggle-row">
      <button
        type="button"
        aria-pressed={checked}
        onClick={() =>
          onChange({
            target: {
              name,
              type: "checkbox",
              checked: !checked,
            },
          })
        }
        className={`toggle ${checked ? "on" : ""}`}
      >
        <span />
      </button>

      <span>{label}</span>
    </label>
  );
}

function DecisionRow({
  icon: Icon,
  label,
  value,
  variant = "purple",
  large = false,
}) {
  return (
    <div className={`decision-row ${large ? "large" : ""}`}>
      <div className={`small-orb small-orb-${variant}`}>
        <Icon size={22} strokeWidth={1.8} />
      </div>

      <div className="decision-row-content">
        <span>{label}</span>

        <div className={`decision-value ${variant}`}>
          <strong>{value}</strong>
          <ChevronRight size={20} />
        </div>
      </div>
    </div>
  );
}

function ConfidenceCard({ label, value, cyan }) {
  return (
    <div className="confidence-card">
      <div className="confidence-label">
        <span>
          {label}
          <Info size={14} />
        </span>

        <strong>{value}%</strong>
      </div>

      <div className="confidence-track">
        <span
          className={cyan ? "cyan" : "purple"}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);

  const update = (e) => {
    const { name, value, checked, type } = e.target;

    setForm((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const clear = () => {
    setForm(initialForm);
  };

  const classify = () => {
    setLoading(true);

    window.setTimeout(() => {
      setLoading(false);
    }, 1300);
  };

  return (
    <main className="supportflow-app">
      <div className="background-light background-light-one" />
      <div className="background-light background-light-two" />
      <div className="background-light background-light-three" />

      <div className="main-glass-shell">
        {/* HEADER */}

        <header className="header">
          <div className="brand">
            <div className="logo-symbol">
              <span />
              <span />
            </div>

            <strong>
              SupportFlow <b>AI</b>
            </strong>
          </div>

          <div className="models-online">
            <span />
            Models online
          </div>

          <label className="top-search">
            <Search size={18} />

            <input
              type="text"
              placeholder="Search tickets, customers, or anything..."
            />
          </label>

          <button className="header-icon notification">
            <Bell size={21} />
            <span />
          </button>

          <button className="profile-button">
            <span className="profile-avatar">JD</span>
            <ChevronDown size={16} />
          </button>
        </header>

        {/* APP BODY */}

        <div className="workspace">
          {/* SIDEBAR */}

          <aside className="sidebar">
            <nav className="sidebar-top">
              <SidebarItem icon={LayoutDashboard} active>
                Dashboard
              </SidebarItem>

              <SidebarItem icon={PlusSquare}>New Ticket</SidebarItem>

              <SidebarItem icon={List} badge={12}>
                Review Queue
              </SidebarItem>

              <SidebarItem icon={BarChart3}>Analytics</SidebarItem>
            </nav>

            <div className="sidebar-spacer" />

            <nav className="sidebar-bottom-links">
              <SidebarItem icon={Settings}>Settings</SidebarItem>
              <SidebarItem icon={CircleHelp}>Help</SidebarItem>
            </nav>

            <div className="sidebar-promo">
              <p>
                Smarter
                <br />
                support
                <br />
                for brighter
                <br />
                experiences.
              </p>

              <div className="promo-orb">
                <div className="promo-orb-reflection" />
              </div>
            </div>

            <div className="large-liquid-orb">
              <div className="large-orb-highlight" />
              <div className="large-orb-ring" />
            </div>
          </aside>

          {/* MAIN */}

          <section className="dashboard-area">
            {/* METRICS */}

            <div className="metric-grid">
              <MetricCard
                icon={Folder}
                title="Category Routing"
                value="93.23%"
              />

              <MetricCard
                icon={Zap}
                title="Priority Accuracy"
                value="97.45%"
                variant="cyan"
              />

              <MetricCard
                icon={Database}
                title="Selective Priority"
                value="98.97%"
              />
            </div>

            {/* LOWER CONTENT */}

            <div className="content-grid">
              {/* FORM */}

              <section className="request-card glass-panel">
                <div className="card-heading">
                  <GlassIcon icon={FileText} />

                  <div>
                    <h1>New Support Request</h1>
                    <p>
                      Describe the issue and let AI classify and prioritize it.
                    </p>
                  </div>
                </div>

                <label className="description-field">
                  <span>
                    Description <b>*</b>
                  </span>

                  <textarea
                    name="description"
                    maxLength={1000}
                    value={form.description}
                    onChange={update}
                    placeholder="Describe the issue, error message, or customer request..."
                  />

                  <small>{form.description.length}/1000</small>
                </label>

                <div className="form-grid">
                  <SelectBox
                    icon={Users}
                    label="Customers affected"
                    name="affected"
                    value={form.affected}
                    onChange={update}
                    required
                    options={[
                      "1",
                      "2 - 10",
                      "11 - 50",
                      "51 - 100",
                      "100+",
                    ]}
                  />

                  <SelectBox
                    icon={Clock3}
                    label="Estimated downtime"
                    name="downtime"
                    value={form.downtime}
                    onChange={update}
                    options={[
                      "No downtime",
                      "Under 15 minutes",
                      "15 - 30 minutes",
                      "30 - 60 minutes",
                      "1 - 4 hours",
                      "4+ hours",
                    ]}
                  />

                  <SelectBox
                    icon={SlidersHorizontal}
                    label="Error rate"
                    name="errorRate"
                    value={form.errorRate}
                    onChange={update}
                    options={[
                      "0% (no errors)",
                      "Under 1%",
                      "1% - 5%",
                      "5% - 20%",
                      "20%+",
                    ]}
                  />

                  <SelectBox
                    icon={Box}
                    label="Product area"
                    name="productArea"
                    value={form.productArea}
                    onChange={update}
                    required
                    options={[
                      "Select product area",
                      "Hardware",
                      "Access",
                      "Storage",
                      "HR Support",
                      "Purchase",
                      "Internal Project",
                    ]}
                  />

                  <SelectBox
                    icon={Crown}
                    label="Customer tier"
                    name="customerTier"
                    value={form.customerTier}
                    onChange={update}
                    options={[
                      "Standard",
                      "Business",
                      "Enterprise",
                      "VIP",
                    ]}
                  />
                </div>

                <div className="impact-section">
                  <div className="impact-heading">
                    <strong>
                      Impact indicators
                      <Info size={15} />
                    </strong>
                  </div>

                  <div className="toggle-grid">
                    <Toggle
                      name="serviceDisruption"
                      label="Service disruption"
                      checked={form.serviceDisruption}
                      onChange={update}
                    />

                    <Toggle
                      name="securityRelated"
                      label="Security related"
                      checked={form.securityRelated}
                      onChange={update}
                    />

                    <Toggle
                      name="revenueImpact"
                      label="Revenue impact"
                      checked={form.revenueImpact}
                      onChange={update}
                    />

                    <Toggle
                      name="vipCustomer"
                      label="VIP customer"
                      checked={form.vipCustomer}
                      onChange={update}
                    />
                  </div>
                </div>

                <div className="form-actions">
                  <button onClick={clear} className="clear-button">
                    Clear
                  </button>

                  <button onClick={classify} className="classify-button">
                    {loading ? (
                      <>
                        <span className="spinner" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Sparkles size={20} />
                        Classify with AI
                        <ChevronRight size={20} />
                      </>
                    )}
                  </button>
                </div>
              </section>

              {/* AI DECISION */}

              <section className="decision-card glass-panel">
                <div className="decision-heading">
                  <div className="decision-heading-title">
                    <GlassIcon icon={BrainCircuit} />

                    <div>
                      <h2>ML Decision</h2>
                      <p>AI analysis based on your input</p>
                    </div>
                  </div>

                  <div className="automatic-badge">
                    <span className="check-orb">
                      <Check size={18} />
                    </span>

                    Automatic decision

                    <Info size={16} />
                  </div>
                </div>

                <div className="decision-details">
                  <DecisionRow
                    icon={Database}
                    label="Predicted category"
                    value="Hardware"
                  />

                  <DecisionRow
                    icon={Users}
                    label="Assigned team"
                    value="Hardware Support Team"
                  />

                  <div className="priority-response-grid">
                    <DecisionRow
                      icon={Star}
                      label="Priority"
                      value="HIGH"
                      variant="danger"
                    />

                    <DecisionRow
                      icon={Clock3}
                      label="Response target"
                      value="30 minutes"
                      variant="cyan"
                    />
                  </div>
                </div>

                <div className="confidence-grid">
                  <ConfidenceCard
                    label="Category confidence"
                    value={96}
                  />

                  <ConfidenceCard
                    label="Priority confidence"
                    value={89}
                    cyan
                  />
                </div>

                <div className="decision-explanation">
                  <div className="explanation-icon">
                    <Sparkles size={21} />
                  </div>

                  <p>
                    This ticket was automatically classified and prioritized by
                    SupportFlow AI using our latest model.
                  </p>

                  <div className="explanation-divider" />

                  <p className="explanation-right">
                    <strong>High confidence</strong>
                    Match found in similar tickets
                  </p>
                </div>
              </section>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}