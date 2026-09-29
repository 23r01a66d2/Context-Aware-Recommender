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
  getClient: (clientId) => request(`/clients/${clientId}`),
  createClient: (payload) => request('/clients', {
    method: 'POST',
    body: JSON.stringify(payload),
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

  // 6. Model Registry
  getModels: (clientId) => request(`/clients/${clientId}/models`),
  getModelVersion: (clientId, versionTag) => request(`/clients/${clientId}/models/${versionTag}`),
  activateModel: (clientId, versionTag) => request(`/clients/${clientId}/models/${versionTag}/activate`, {
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
};
