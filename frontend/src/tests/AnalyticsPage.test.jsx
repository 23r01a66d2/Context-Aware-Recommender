import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import AnalyticsPage from '../pages/admin/AnalyticsPage';
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
if (typeof global !== 'undefined' && !global.ResizeObserver) {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

vi.mock('../services/api', () => ({
  api: {
    getModels: vi.fn(),
    getModelAnalytics: vi.fn(),
    getClients: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ThemeProvider>
        <ClientProvider>{ui}</ClientProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

describe('AnalyticsPage Model Benchmark Tests', () => {
  const mockClients = [
    { client_id: 'demo_ecommerce', name: 'Indian E-Commerce Benchmark' },
    { client_id: 'empty_client', name: 'Empty Client' },
  ];

  const mockModels = [
    {
      model_id: 'demo_ecommerce_v1',
      version_tag: 'v1',
      is_active: true,
      cold_start_threshold: 3,
      metrics: {
        precision_at_5: 0.0121,
        recall_at_5: 0.0603,
        ndcg_at_5: 0.0352,
        hit_rate_at_5: 0.0603,
      },
    },
  ];

  const mockAnalyticsData = {
    client_id: 'demo_ecommerce',
    client_name: 'Indian E-Commerce Benchmark',
    models_comparison: [
      {
        model_id: 'model_1',
        name: 'Model 1: Popularity Baseline',
        approach: 'Frequency / Popularity Heuristic',
        architecture: 'Item Frequency Ranking',
        status: 'Evaluated Baseline',
        is_active: false,
        k: 5,
        precision_at_5: 0.0106,
        precision_formatted: '1.06%',
        recall_at_5: 0.053,
        ndcg_at_5: 0.032,
        hit_rate_at_5: 0.053,
        mrr: 0.0215,
        evaluated_cases: 1377,
        relevance_definition: '1 positive purchase + 99 unpurchased negatives per test session',
      },
      {
        model_id: 'model_2',
        name: 'Model 2: Content-Based Profile',
        approach: 'Item Attribute Cosine Match',
        architecture: 'Profile Centroid Cosine Match',
        status: 'Evaluated Baseline',
        is_active: false,
        k: 5,
        precision_at_5: 0.0298,
        precision_formatted: '2.98%',
        recall_at_5: 0.1489,
        ndcg_at_5: 0.1374,
        hit_rate_at_5: 0.1489,
        mrr: 0.0782,
        evaluated_cases: 1377,
      },
      {
        model_id: 'model_3',
        name: 'Model 3: Collaborative Filtering (SVD)',
        approach: 'Matrix Factorization (MF)',
        architecture: 'Latent Matrix Factorization',
        status: 'Evaluated Baseline',
        is_active: false,
        k: 5,
        precision_at_5: 0.0109,
        precision_formatted: '1.09%',
        recall_at_5: 0.0545,
        ndcg_at_5: 0.0308,
        hit_rate_at_5: 0.0545,
        mrr: 0.0241,
        evaluated_cases: 1377,
      },
      {
        model_id: 'model_4',
        name: 'Model 4: Static Multi-Modal Fusion',
        approach: 'Two-Tower Static Weighting',
        architecture: 'Two-Tower Fixed Weights',
        status: 'Evaluated Architecture',
        is_active: false,
        k: 5,
        precision_at_5: 0.0129,
        precision_formatted: '1.29%',
        recall_at_5: 0.0646,
        ndcg_at_5: 0.0401,
        hit_rate_at_5: 0.0646,
        mrr: 0.0289,
        evaluated_cases: 1377,
      },
      {
        model_id: 'model_5',
        name: 'Model 5: Proposed Adaptive Multi-Modal (v1)',
        approach: 'Adaptive Representation Fusion + Cold Gate',
        architecture: 'Two-Tower with Point-in-time Cold Gate',
        status: 'Active Target',
        is_active: true,
        k: 5,
        precision_at_5: 0.0121,
        precision_formatted: '1.21%',
        recall_at_5: 0.0603,
        ndcg_at_5: 0.0352,
        hit_rate_at_5: 0.0603,
        mrr: 0.0895,
        evaluated_cases: 1377,
      },
      {
        model_id: 'demo_ecommerce_v2',
        name: 'Indian E-Commerce Recommender (v2)',
        approach: 'Two-Tower Adaptive Fusion',
        architecture: 'Two-Tower Candidate Architecture',
        status: 'Candidate Model',
        is_active: false,
        k: 5,
        precision_at_5: 0.20,
        precision_formatted: '20.00%',
        recall_at_5: 1.0,
        ndcg_at_5: 0.8589,
        hit_rate_at_5: 1.0,
        mrr: 0.633,
        evaluated_cases: 100,
      },
      {
        model_id: 'unevaluated_model',
        name: 'Experimental Heuristic (Unranked)',
        approach: 'Experimental Heuristic',
        architecture: 'No offline ranking dataset available',
        status: 'Trained (No Ranking Test Set)',
        is_active: false,
        k: 5,
        precision_at_k: null,
        precision_formatted: 'N/A',
        evaluated_cases: null,
        unavailable_reason: 'Candidate ranking evaluation set has not been generated.',
      },
    ],
    visual_models_comparison: [
      {
        model_id: 'mlvis_statistical_baseline',
        name: 'Statistical Baseline (Heuristic)',
        approach: 'Column / Dataset Profile Heuristics',
        status: 'Evaluated Baseline',
        k: 3,
        precision_at_k: 0.6245,
        precision_formatted: '62.45%',
        precision_at_3: 0.6245,
        precision_at_1: 0.3972,
        accuracy: 0.3972,
        macro_f1: 0.1824,
        ndcg_at_3: 0.5169,
        evaluated_cases: 783,
        relevance_definition: 'True chart type ranked in top 3 of 6 chart families (kdd21-MLVis 1K benchmark)',
      },
      {
        model_id: 'mlvis_v1',
        name: 'MLVis Neural Recommender (v1.0.0)',
        approach: 'Deep Feedforward Neural Classifier',
        status: 'Production Default',
        is_active: true,
        k: 3,
        precision_at_k: 0.9987,
        precision_formatted: '99.87%',
        precision_at_3: 0.9987,
        precision_at_1: 0.9068,
        accuracy: 0.9068,
        macro_f1: 0.7722,
        ndcg_at_3: 0.9614,
        evaluated_cases: 783,
        relevance_definition: 'True chart type ranked in top 3 of 6 chart families (kdd21-MLVis 1K benchmark)',
      },
      {
        model_id: 'mlvis_v2',
        name: 'MLVis Recommender v2 (Candidate Release)',
        approach: 'Synthetic Multi-Task Augmentation',
        status: 'Candidate Evaluated',
        is_active: false,
        k: 3,
        precision_at_k: 0.9898,
        precision_formatted: '98.98%',
        precision_at_3: 0.9898,
        precision_at_1: 0.9093,
        accuracy: 0.9093,
        macro_f1: 0.7712,
        ndcg_at_3: 0.9572,
        evaluated_cases: 783,
        relevance_definition: 'True chart type ranked in top 3 of 6 chart families (kdd21-MLVis 1K benchmark)',
      },
    ],
    modality_variants: [
      { variant: 'Text Only', precision_at_5: 0.021, recall_at_5: 0.105, ndcg_at_5: 0.098 },
      { variant: 'Image + Text', precision_at_5: 0.0285, recall_at_5: 0.138, ndcg_at_5: 0.126 },
      { variant: 'Full Multimodal + Context', precision_at_5: 0.0345, recall_at_5: 0.162, ndcg_at_5: 0.145 },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    api.getClients.mockResolvedValue(mockClients);
    api.getModels.mockResolvedValue(mockModels);
    api.getModelAnalytics.mockResolvedValue(mockAnalyticsData);
  });

  it('renders page header, title, and essential controls', async () => {
    renderWithProviders(<AnalyticsPage />);

    expect(screen.getByText(/Model Analytics & Architectural Benchmark/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText(/Product Recommendation Models/i)).toBeInTheDocument();
      expect(screen.getByText(/MLVis Visualization Recommenders/i)).toBeInTheDocument();
      expect(screen.getByText(/Precision Protocol Guide/i)).toBeInTheDocument();
    });
  });

  it('renders Precision@5 as primary metric with percentage formatting and no 0.000', async () => {
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText('Precision@5 (Primary)')).toBeInTheDocument();
      // Verifies actual percentages are displayed, not zero
      expect(screen.getByText('1.06%')).toBeInTheDocument();
      expect(screen.getByText('2.98%')).toBeInTheDocument();
      expect(screen.getByText('1.09%')).toBeInTheDocument();
      expect(screen.getByText('1.29%')).toBeInTheDocument();
      expect(screen.getByText('1.21%')).toBeInTheDocument();
      expect(screen.getByText('20.00%')).toBeInTheDocument();
    });
  });

  it('renders dynamic models beyond the 5 hardcoded limit', async () => {
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      // 7 models in mock data
      expect(screen.getByText('Discovered Models:')).toBeInTheDocument();
      expect(screen.getByText('7')).toBeInTheDocument();
      expect(screen.getByText('Indian E-Commerce Recommender (v2)')).toBeInTheDocument();
      expect(screen.getByText('Experimental Heuristic (Unranked)')).toBeInTheDocument();
    });
  });

  it('displays N/A for unevaluated models without misleading zeros and expands explanation', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText('Experimental Heuristic (Unranked)')).toBeInTheDocument();
    });

    // Precision is N/A
    const naBadges = screen.getAllByText('N/A');
    expect(naBadges.length).toBeGreaterThanOrEqual(1);

    // Expand the details row
    const expandButtons = screen.getAllByTitle(/Expand full metrics/i);
    expect(expandButtons.length).toBeGreaterThan(0);
    await user.click(expandButtons[expandButtons.length - 1]);

    await waitFor(() => {
      expect(screen.getByText(/Metric Limitation:/i)).toBeInTheDocument();
      expect(screen.getByText(/Candidate ranking evaluation set has not been generated/i)).toBeInTheDocument();
    });
  });

  it('allows expanding a model row to view secondary metrics and relevance definition', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText('Model 1: Popularity Baseline')).toBeInTheDocument();
    });

    const expandButtons = screen.getAllByTitle(/Expand full metrics/i);
    await user.click(expandButtons[0]);

    await waitFor(() => {
      expect(screen.getByText(/Relevance Definition:/i)).toBeInTheDocument();
      expect(screen.getByText(/1 positive purchase \+ 99 unpurchased negatives/i)).toBeInTheDocument();
    });
  });

  it('allows toggling to MLVis Visualization Recommendation benchmark with Top-3 protocol', async () => {
    const user = userEvent.setup();
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/MLVis Visualization Recommenders/i)).toBeInTheDocument();
    });

    const visualTabBtn = screen.getByText(/MLVis Visualization Recommenders/i);
    await user.click(visualTabBtn);

    await waitFor(() => {
      expect(screen.getByText('Precision@3 (Primary)')).toBeInTheDocument();
      expect(screen.getByText('MLVis Neural Recommender (v1.0.0)')).toBeInTheDocument();
      expect(screen.getByText('99.87%')).toBeInTheDocument();
      expect(screen.getByText('62.45%')).toBeInTheDocument();
      expect(screen.getByText('98.98%')).toBeInTheDocument();
      expect(screen.getAllByText(/charts/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders clean empty state when tenant has no models registered', async () => {
    api.getModelAnalytics.mockResolvedValueOnce({
      client_id: 'empty_client',
      client_name: 'Empty Client',
      models_comparison: [],
      visual_models_comparison: [],
    });

    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/No Evaluation Benchmark Data Available/i)).toBeInTheDocument();
      expect(screen.getByText(/Go to Model Training/i)).toBeInTheDocument();
    });
  });

  it('handles API error gracefully without crashing', async () => {
    api.getModelAnalytics.mockRejectedValueOnce(new Error('Network connectivity issue'));
    renderWithProviders(<AnalyticsPage />);

    await waitFor(() => {
      expect(screen.getByText(/Network connectivity issue/i)).toBeInTheDocument();
    });
  });
});
