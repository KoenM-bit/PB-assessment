import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ListingsPage } from "./pages/ListingsPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { PredictPage } from "./pages/PredictPage";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<PredictPage />} />
          <Route path="listings" element={<ListingsPage />} />
          <Route path="actual-sale" element={<Navigate to="/listings" replace />} />
          <Route path="monitoring" element={<MonitoringPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
