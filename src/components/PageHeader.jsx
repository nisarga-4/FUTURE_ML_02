export default function PageHeader({ eyebrow='SUPPORTFLOW AI', title, text, actions }) {
  return (
    <div className="page-header">
      <div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{text}</p></div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}
