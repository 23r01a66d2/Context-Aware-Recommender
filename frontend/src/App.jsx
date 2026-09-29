import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ClientProvider } from './context/ClientContext';
import AdminLayout from './layouts/AdminLayout';
import UserLayout from './layouts/UserLayout';

import DashboardPage from './pages/admin/DashboardPage';
import ClientsPage from './pages/admin/ClientsPage';
import NewClientPage from './pages/admin/NewClientPage';
import DatasetUploadPage from './pages/admin/DatasetUploadPage';
import SchemaMappingPage from './pages/admin/SchemaMappingPage';
import TrainingPage from './pages/admin/TrainingPage';
import ModelsPage from './pages/admin/ModelsPage';
import AnalyticsPage from './pages/admin/AnalyticsPage';
import FeedbackPage from './pages/admin/FeedbackPage';
import RecommendationPage from './pages/user/RecommendationPage';

export default function App() {
  return (
    <BrowserRouter>
      <ClientProvider>
        <Routes>
          {/* Default Root Redirects to Admin */}
          <Route path="/" element={<Navigate to="/admin" replace />} />

          {/* Admin Portal Layout */}
          <Route path="/admin" element={<AdminLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="clients" element={<ClientsPage />} />
            <Route path="clients/new" element={<NewClientPage />} />
            <Route path="clients/:clientId/dataset" element={<DatasetUploadPage />} />
            <Route path="clients/:clientId/schema" element={<SchemaMappingPage />} />
            <Route path="clients/:clientId/training" element={<TrainingPage />} />
            <Route path="clients/:clientId/models" element={<ModelsPage />} />
            <Route path="clients/:clientId/analytics" element={<AnalyticsPage />} />
            <Route path="clients/:clientId/feedback" element={<FeedbackPage />} />
          </Route>

          {/* Live User Experience Layout */}
          <Route path="/recommend" element={<UserLayout />}>
            <Route index element={<RecommendationPage />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/admin" replace />} />
        </Routes>
      </ClientProvider>
    </BrowserRouter>
  );
}
