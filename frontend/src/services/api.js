/**
 * Centralized API Service for FastAPI Multi-Modal Recommender Platform
 * Connects directly to backend on VITE_API_BASE_URL (defaults to /api via Vite proxy or http://localhost:8000/api)
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const headers = { ...options.headers };

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  const contentType = response.headers.get('content-type');
  let data = null;
  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const errorMsg = data?.detail || data?.message || (typeof data === 'string' ? data : `HTTP ${response.status}: ${response.statusText}`);
    const error = new Error(typeof errorMsg === 'object' ? JSON.stringify(errorMsg) : errorMsg);
    error.status = response.status;
    error.data = data;
    throw error;
  }

  return data;
}

export const api = {
  // 1. Health
  getHealth: () => request('/health'),

  // 2. Clients
  getClients: async () => {
    const res = await request('/clients');
    return Array.isArray(res) ? res : (res?.clients || []);
  },
  getClientsOverview: async () => {
    const res = await request('/clients/overview');
    return res?.domains || [];
  },
  getClient: (clientId) => request(`/clients/${clientId}`),
  createClient: (payload) => request('/clients', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  deleteClient: (clientId, deletePhysicalFiles = false) => request(`/clients/${clientId}?delete_physical_files=${deletePhysicalFiles}`, {
    method: 'DELETE',
  }),

  // 3. Datasets
  getDataset: (clientId) => request(`/clients/${clientId}/dataset`),
  getDatasetPreview: (clientId, limit = 10, offset = 0) =>
    request(`/clients/${clientId}/dataset/preview?limit=${limit}&offset=${offset}`),
  uploadDataset: (clientId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request(`/clients/${clientId}/dataset`, {
      method: 'POST',
      body: formData,
    });
  },

  // 4. Schema Mapping
  getSchema: (clientId) => request(`/clients/${clientId}/schema`),
  saveSchema: (clientId, payload) => request(`/clients/${clientId}/schema`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  // 5. Training Lifecycle
  getTrainingStatus: (clientId) => request(`/clients/${clientId}/training-status`),
  triggerTraining: (clientId, payload = {}) => request(`/clients/${clientId}/train`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  // 6. Model Registry & Analytics
  getModels: (clientId) => request(`/clients/${clientId}/models`),
  getAllModels: () => request('/models'),
  getModelAnalytics: (clientId) => request(clientId ? `/models/analytics?client_id=${encodeURIComponent(clientId)}` : '/models/analytics'),
  getModelVersion: (clientId, versionTag) => request(`/clients/${clientId}/models/${versionTag}`),
  activateModel: (clientId, versionTag) => request(`/clients/${clientId}/models/${versionTag}/activate`, {
    method: 'POST',
  }),

  // Schema Auto-Repair
  autoRepairSchema: (clientId) => request(`/clients/${clientId}/schema/auto-repair`, {
    method: 'POST',
  }),

  // 7. Live Real-Time Recommendations
  getRecommendations: (payload) => request('/recommend', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  // 8. Feedback Audit Log
  recordFeedback: (payload) => request('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  getFeedbackLog: (clientId) => request(`/feedback${clientId ? `?client_id=${clientId}` : ''}`),

  // 9. Visual Analytics & Interactive Recommendations
  getVisualDatasets: () => request('/visual-analytics/datasets'),
  uploadVisualDataset: (file, clientId, userId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams();
    if (clientId) params.append('client_id', clientId);
    if (userId) params.append('user_id', userId);
    const query = params.toString() ? `?${params.toString()}` : '';
    return request(`/visual-analytics/upload${query}`, {
      method: 'POST',
      body: formData,
    });
  },

  getDatasetProfile: (datasetId = 'demo_ecommerce', clientId = null) => {
    const params = new URLSearchParams({ dataset_id: datasetId });
    if (clientId) params.append('client_id', clientId);
    return request(`/visual-analytics/profile?${params.toString()}`);
  },
  getChartData: (payload) => request('/visual-analytics/chart-data', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  getVisualRecommendations: (payload) => request('/visual-analytics/recommend', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  recordVisualInteraction: (payload) => request('/visual-analytics/interactions', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  getVisualInteractions: (sessionId = null, clientId = null) => {
    const params = new URLSearchParams();
    if (sessionId) params.append('session_id', sessionId);
    if (clientId) params.append('client_id', clientId);
    const q = params.toString() ? `?${params.toString()}` : '';
    return request(`/visual-analytics/interactions${q}`);
  },
  getVisualAnalytics: (clientId = null) => {
    const q = clientId ? `?client_id=${encodeURIComponent(clientId)}` : '';
    return request(`/visual-analytics/analytics${q}`);
  },

  // 10. Persistent User History & Cold/Warm Start Detection
  getUserHistoryStatus: (clientId, userId) => {
    const params = new URLSearchParams({
      client_id: clientId,
      user_id: userId,
    });
    return request(`/user-history/status?${params.toString()}`);
  },
  getUserHistory: (clientId, userId, limit = 50, offset = 0) => {
    const params = new URLSearchParams({
      client_id: clientId,
      user_id: userId,
      limit: String(limit),
      offset: String(offset),
    });
    return request(`/user-history?${params.toString()}`);
  },
  recordUserHistory: (payload) => request('/user-history/record', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
};


