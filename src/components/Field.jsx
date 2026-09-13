export function SelectField({ label, name, value, onChange, options, required, error }) {
  return (
    <label className="field">
      <span>{label}{required && <em>*</em>}</span>
      <select name={name} value={value} onChange={onChange} aria-invalid={!!error}>
        <option value="">Select option</option>
        {options.map(o => <option key={o}>{o}</option>)}
      </select>
      {error && <small className="error-text">{error}</small>}
    </label>
  )
}

export function Switch({ label, name, checked, onChange }) {
  return (
    <label className="switch-row">
      <span>{label}</span>
      <input type="checkbox" name={name} checked={checked} onChange={onChange}/>
      <span className="switch-ui" aria-hidden="true"><i/></span>
    </label>
  )
}
