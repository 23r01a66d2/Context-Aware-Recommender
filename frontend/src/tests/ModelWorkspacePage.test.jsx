import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import ModelsPage from '../pages/admin/ModelsPage';
import ModelWorkspacePage from '../pages/admin/ModelWorkspacePage';
import { ClientProvider } from '../context/ClientContext';
import { ThemeProvider } from '../context/ThemeContext';
import { api } from '../services/api';

if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

vi.mock('../services/api', () => ({
  api: {
    getModels: vi.fn(),
    getAllModels: vi.fn(),
    getModelVersion: vi.fn(),
    activateModel: vi.fn(),
    getClients: vi.fn(),
    getVisualDatasets: vi.fn(),
    getDatasetProfile: vi.fn(),
    getChartData: vi.fn(),
    getVisualRecommendations: vi.fn(),
    recordVisualInteraction: vi.fn(),
    getVisualInteractions: vi.fn(),
    getVisualAnalytics: vi.fn(),
  },
}));

describe('ModelsPage & ModelWorkspacePage Integration Tests', () => {
  const mockGlobalModels = [
    {
      model_id: 'mlvis_v1',
      client_id: 'mlvis',
      client_name: 'MLVis Visualization Suite',
      name: 'MLVis Neural Visualization Recommender',
      model_type: 'MLVis Neural Recommender',
      supported_task: 'Visual Analytics & Chart Recommendation',
      supports_visualizations: true,
      version: '1.0.0',
      version_tag: 'v1.0.0',
      architecture: 'Deep Feedforward Neural Classifier (26 meta-features -> 6 chart types)',
      modalities: ['Dataset Meta-Features', 'Column Encodings', 'Statistical Profiles'],
      training_status: 'Production Ready',
      is_active: true,
      metrics: {
        accuracy: 0.9067,
        macro_f1: 0.7721,
        precision_at_3: 0.9987,
        ndcg_at_3: 0.9614,
      },
      eval_report: {
        mlvis_neural_model: {
          accuracy: 0.9067,
          macro_f1: 0.7721,
          precision_at_3: 0.9987,
          ndcg_at_3: 0.9614,
        },
        statistical_baseline: {
          accuracy: 0.3971,
          macro_f1: 0.1824,
        },
      },
      feature_dimensions: { meta_features: 26, chart_classes: 6 },
    },
    {
      model_id: 'demo_ecommerce_v1',
      client_id: 'demo_ecommerce',
      client_name: 'Indian E-Commerce Benchmark',
      name: 'Indian E-Commerce Benchmark Recommender (v1)',
      model_type: 'Two-Tower Recommender',
      supported_task: 'Product / Item Recommendation',
      supports_visualizations: false,
      version: 'v1',
      version_tag: 'v1',
      architecture: 'Two-Tower Adaptive Fusion with Cold-Start Gating',
      modalities: ['Behavioral History', 'Catalog Metadata', 'Session Context'],
      training_status: 'Active Target',
      is_active: true,
      metrics: {
        ndcg_at_5: 0.443,
        hit_rate_at_5: 0.521,
      },
      feature_dimensions: { behavior: 16, content: 12, context: 21 },
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    api.getAllModels.mockResolvedValue(mockGlobalModels);
    api.getModels.mockResolvedValue([mockGlobalModels[1]]);
    api.getClients.mockResolvedValue([
      { client_id: 'demo_ecommerce', name: 'Indian E-Commerce Benchmark' },
    ]);
    api.getVisualDatasets.mockResolvedValue([
      { id: 'demo_ecommerce', name: 'Indian E-Commerce Benchmark', filename: 'Ecommerce.csv' },
    ]);
    api.getDatasetProfile.mockResolvedValue({
      dimensions: { total_rows: 25000, total_columns: 29 },
      filename: 'Ecommerce.csv',
      columns: [{ name: 'unit_price', dtype: 'numeric', stats: { mean: 250 } }],
      numeric_columns: ['unit_price'],
      categorical_columns: ['product_category'],
      correlations: { pairs: [] },
    });
    api.getChartData.mockResolvedValue({
      chart_type: 'bar',
      data: [{ label: 'cat_1', value: 100 }],
    });
    api.getVisualRecommendations.mockResolvedValue({
      recommendations: [
        {
          candidate_id: 'cand_1',
          chart_type: 'bar',
          recommendation_score: 0.95,
          score_source: 'MLVis Neural Recommender',
          explanation: 'Optimal frequency comparison.',
        },
      ],
      score_source: 'MLVis Neural Recommender',
    });
  });

  it('renders ModelsPage as main discovery portal with MLVis and Two-Tower models', async () => {
    render(
      <MemoryRouter initialEntries={['/admin/models']}>
        <ThemeProvider>
          <ClientProvider>
            <ModelsPage />
          </ClientProvider>
        </ThemeProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Model Registry & Intelligence Workspace/i)).toBeInTheDocument();
      expect(screen.getByText(/MLVis Neural Visualization Recommender/i)).toBeInTheDocument();
      expect(screen.getByText(/Supports Visualizations/i)).toBeInTheDocument();
      expect(screen.getByText(/Two-Tower Adaptive Fusion/i)).toBeInTheDocument();
    });
  });

  it('opens ModelWorkspacePage for MLVis with Overview, Visualizations, and Evaluation tabs', async () => {
    render(
      <MemoryRouter initialEntries={['/admin/models/mlvis/workspace?tab=overview']}>
        <ThemeProvider>
          <ClientProvider>
            <Routes>
              <Route path="/admin/models/:modelId/workspace" element={<ModelWorkspacePage />} />
            </Routes>
          </ClientProvider>
        </ThemeProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Overview & Architecture/i)).toBeInTheDocument();
      expect(screen.getByText(/Visualizations & Analytics/i)).toBeInTheDocument();
      expect(screen.getByText(/Evaluation & Benchmark/i)).toBeInTheDocument();
      expect(screen.getByText(/Visual Analytics & Chart Recommendation/i)).toBeInTheDocument();
    });
  });

  it('opens Visualizations tab on MLVis workspace and mounts Visual Analytics with MLVis locked', async () => {
    render(
      <MemoryRouter initialEntries={['/admin/models/mlvis/workspace?tab=visualizations']}>
        <ThemeProvider>
          <ClientProvider>
            <Routes>
              <Route path="/admin/models/:modelId/workspace" element={<ModelWorkspacePage />} />
            </Routes>
          </ClientProvider>
        </ThemeProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/MLVis Recommender \(Active\)/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Dataset/i)).toBeInTheDocument();
      expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();
    });
  });

  it('shows clear explanation when visualization tab is opened for incompatible Two-Tower product model', async () => {
    render(
      <MemoryRouter initialEntries={['/admin/models/demo_ecommerce/workspace?tab=visualizations']}>
        <ThemeProvider>
          <ClientProvider>
            <Routes>
              <Route path="/admin/models/:modelId/workspace" element={<ModelWorkspacePage />} />
            </Routes>
          </ClientProvider>
        </ThemeProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Visualization Recommendations Unsupported/i)).toBeInTheDocument();
      expect(screen.getByText(/Switch to MLVis Visualization Workspace/i)).toBeInTheDocument();
    });
  });

  it('displays evaluation benchmark metrics in the Evaluation tab for MLVis', async () => {
    render(
      <MemoryRouter initialEntries={['/admin/models/mlvis/workspace?tab=evaluation']}>
        <ThemeProvider>
          <ClientProvider>
            <Routes>
              <Route path="/admin/models/:modelId/workspace" element={<ModelWorkspacePage />} />
            </Routes>
          </ClientProvider>
        </ThemeProvider>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Evaluation Metrics & Performance/i)).toBeInTheDocument();
      expect(screen.getByText(/90.7%/i)).toBeInTheDocument(); // Test accuracy
      expect(screen.getByText(/Top-3 Chart Candidate Match/i)).toBeInTheDocument();
    });
  });
});
