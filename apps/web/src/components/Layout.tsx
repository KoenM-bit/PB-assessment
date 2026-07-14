import { NavLink, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="app">
      <nav className="nav">
        <h1>House Price ML</h1>
        <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
          New Prediction
        </NavLink>
        <NavLink to="/listings" className={({ isActive }) => (isActive ? "active" : "")}>
          Predictions & Sales
        </NavLink>
        <NavLink to="/monitoring" className={({ isActive }) => (isActive ? "active" : "")}>
          Monitoring
        </NavLink>
      </nav>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
