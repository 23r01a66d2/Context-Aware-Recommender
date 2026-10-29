import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import RecommendationPage from '../pages/user/RecommendationPage';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

import { ThemeProvider } from '../context/ThemeContext';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    getRecommendations: vi.fn(),
    recordFeedback: vi.fn(),
    getSchema: vi.fn(),
    getVisualDatasets: vi.fn(),
    uploadVisualDataset: vi.fn(),
    getDatasetProfile: vi.fn(),
    getChartData: vi.fn(),
    getVisualRecommendations: vi.fn(),
    recordVisualInteraction: vi.fn(),
    getVisualInteractions: vi.fn(),
    getVisualAnalytics: vi.fn(),
    getUserHistoryStatus: vi.fn(),
    getUserHistory: vi.fn(),
    recordUserHistory: vi.fn(),
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

describe('RecommendationPage Frontend Tests', () => {
  const mockClients = [
    { client_id: 'demo_ecommerce', name: 'Demo E-Commerce', active_model_version: 'v1' },
    { client_id: 'movielens_dataset', name: 'movielens_dataset', active_model_version: 'v1' },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    api.getHealth.mockResolvedValue({ status: 'ok', service: 'context-aware-recommender' });
    api.getClients.mockResolvedValue(mockClients);
    api.getSchema.mockResolvedValue({ column_mappings: {} });
    api.recordFeedback.mockResolvedValue({ id: 1, status: 'recorded' });
    api.getVisualDatasets.mockResolvedValue([
      { id: 'demo_ecommerce', name: 'Indian E-Commerce Benchmark', filename: 'Ecommerce.csv', is_canonical: true }
    ]);
    api.getDatasetProfile.mockResolvedValue({
      dimensions: {
        total_rows: 100,
        total_columns: 5,
        memory_mb: 1.2,
        numeric_columns_count: 2,
        categorical_columns_count: 3,
        total_missing_values: 0,
      },
      filename: 'Ecommerce.csv',
      columns: [{ name: 'Category', dtype: 'categorical', unique_count: 3, null_count: 0, null_pct: 0 }],
      numeric_columns: ['Price'],
      categorical_columns: ['Category'],
      sample_records: [{ Category: 'Tech', Price: 99 }]
    });
    api.getVisualRecommendations.mockResolvedValue({
      model_type: 'mlvis',
      model_version: 'v1.0.0',
      dataset_id: 'demo_ecommerce',
      recommendations: []
    });
    api.getChartData.mockResolvedValue({
      chart_type: 'bar',
      x_column: 'Category',
      y_column: 'count',
      data: [{ label: 'Tech', value: 10 }]
    });
    api.getVisualInteractions.mockResolvedValue([]);
    api.getVisualAnalytics.mockResolvedValue({
      total_impressions: 10,
      total_clicks: 5,
      ctr_percentage: 50.0
    });
    api.getUserHistoryStatus.mockResolvedValue({
      client_id: 'demo_ecommerce',
      user_id: '9272',
      status: 'warm_start',
      is_cold_start: false,
      history_count: 5,
      last_active: '2026-10-12T10:00:00Z',
      recent_datasets: ['demo_ecommerce']
    });
    api.getUserHistory.mockResolvedValue({
      client_id: 'demo_ecommerce',
      user_id: '9272',
      status: 'warm_start',
      total_records: 1,
      history: [
        { event_id: 'evt_1', entry_type: 'chart_view', dataset_id: 'demo_ecommerce', timestamp: '2026-10-12T10:00:00Z' }
      ]
    });
  });


  // Helper to switch to product recommendation mode in tests that verify product scoring
  const switchToProductMode = async (user) => {
    const prodBtn = screen.getByRole('button', { name: /Product Recommender/i });
    await user.click(prodBtn);
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Score & Rank Candidates/i })).toBeInTheDocument();
    });
  };

  // Test A: Recommendation page renders
  it('A: Recommendation page renders essential controls with MLVis as primary default', async () => {
    const user = userEvent.setup();
    renderWithProviders(<RecommendationPage />);

    expect(screen.getByText(/Live Candidate Scoring & Recommendation Demo/i)).toBeInTheDocument();
    // Default mode is Visual Analytics Recommender (MLVis)
    expect(screen.getByRole('button', { name: /Visual Analytics Recommender \(MLVis\)/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Product Recommender \(Two-Tower\)/i })).toBeInTheDocument();

    // MLVis workspace is shown by default
    expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();

    // Switching to Product Recommender reveals product controls
    await switchToProductMode(user);
    expect(screen.getByText(/Customer \/ User Identifier/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Score & Rank Candidates/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Try New User/i })).toBeInTheDocument();
  });

  // Test B: Cold-start response (history_count = 0, status = cold_start, renders "New User — Cold Start", actual weights)
  it('B: Cold-start response renders "New User — Cold Start" and modality weights', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_cold_001',
      client_id: 'demo_ecommerce',
      user_id: 'cold_user_999',
      status: 'cold_start',
      history_count: 0,
      candidate_pool_size: 899,
      inference_latency_ms: 24.5,
      modality_weights: { behavior: 0.008, content: 0.542, context: 0.450 },
      recommendations: [
        {
          rank: 1,
          item_id: 'prod_442',
          score: 0.8842,
          score_type: 'sigmoid_score',
          explanation: 'Cold-start prioritization via contextual and content affinity.',
          item_metadata: { product_category: 'Electronics', unit_price: 149.99 },
        },
      ],
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    const submitBtn = screen.getByRole('button', { name: /Score & Rank Candidates/i });
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/New User — Cold Start/i)).toBeInTheDocument();
    });

    // Verify modality weights rendered
    expect(screen.getByText(/Adaptive Representation Fusion/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.008/)).toBeInTheDocument();
    expect(screen.getByText(/0\.542/)).toBeInTheDocument();
    expect(screen.getByText(/0\.450/)).toBeInTheDocument();
  });

  // Test C: Warm-start response (history_count >= threshold, status = warm_start, renders "Warm Start")
  it('C: Warm-start response renders "Warm Start"', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_warm_002',
      client_id: 'demo_ecommerce',
      user_id: '9272',
      status: 'warm_start',
      history_count: 8,
      candidate_pool_size: 899,
      inference_latency_ms: 26.1,
      modality_weights: { behavior: 0.45, content: 0.35, context: 0.20 },
      recommendations: [
        {
          rank: 1,
          item_id: 'prod_101',
          score: 0.9125,
          score_type: 'sigmoid_score',
          explanation: 'Personalized from previous historical customer purchase patterns.',
          item_metadata: { product_category: 'Clothing', unit_price: 49.99 },
        },
      ],
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    const submitBtn = screen.getByRole('button', { name: /Score & Rank Candidates/i });
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/Warm Start/i)).toBeInTheDocument();
    });
  });

  // Test D: Recommendation card displays rank, item ID, Recommendation Score, explanation
  it('D: Recommendation card displays rank, item ID, Recommendation Score, explanation', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_card_003',
      client_id: 'demo_ecommerce',
      user_id: 'user_456',
      status: 'warm_start',
      history_count: 5,
      inference_latency_ms: 25.0,
      modality_weights: { behavior: 0.4, content: 0.4, context: 0.2 },
      recommendations: [
        {
          rank: 1,
          item_id: 'SKU_9988',
          score: 0.8765,
          score_type: 'sigmoid_score',
          explanation: 'Grounded explanation test line.',
          item_metadata: { product_category: 'Kitchen', unit_price: 29.95 },
        },
      ],
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText('#1')).toBeInTheDocument();
      expect(screen.getByText(/Product SKU_9988/i)).toBeInTheDocument();
      expect(screen.getByText(/Recommendation Score:/i)).toBeInTheDocument();
      expect(screen.getByText('0.8765')).toBeInTheDocument();
      expect(screen.getByText('Grounded explanation test line.')).toBeInTheDocument();
    });
  });

  // Test E: Confirm UI does NOT call the score: purchase probability, probability of purchase, chance of purchase
  it('E: UI contains zero instances of misleading probability terminology', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_sem_004',
      client_id: 'demo_ecommerce',
      user_id: 'user_777',
      status: 'cold_start',
      history_count: 0,
      inference_latency_ms: 22.0,
      modality_weights: { behavior: 0.01, content: 0.5, context: 0.49 },
      recommendations: [
        {
          rank: 1,
          item_id: 'prod_777',
          score: 0.8123,
          score_type: 'sigmoid_score',
          explanation: 'Grounded explanation',
          item_metadata: { product_category: 'Books', unit_price: 19.99 },
        },
      ],
    });

    const { container } = renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText('0.8123')).toBeInTheDocument();
    });

    const pageText = container.textContent.toLowerCase();
    expect(pageText).not.toContain('purchase probability');
    expect(pageText).not.toContain('probability of purchase');
    expect(pageText).not.toContain('chance of purchase');
    expect(pageText).not.toContain('chance you will buy');
    expect(pageText).not.toContain('sigmoid_probability');
  });

  // Test F: Feedback action sends recommendation_id, client_id, user_id, item_id, action
  it('F: Natural interaction submits correct payload with recommendation_id linkage', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_fb_link_005',
      client_id: 'demo_ecommerce',
      user_id: 'user_fb_test',
      status: 'warm_start',
      history_count: 4,
      inference_latency_ms: 23.0,
      modality_weights: { behavior: 0.3, content: 0.4, context: 0.3 },
      recommendations: [
        {
          rank: 1,
          item_id: 'prod_fb_target',
          score: 0.85,
          score_type: 'sigmoid_score',
          explanation: 'Explanation line',
          item_metadata: { product_category: 'Sports', unit_price: 89.0 },
        },
      ],
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Product prod_fb_target/i)).toBeInTheDocument();
    });

    const viewBtn = screen.getByRole('button', { name: /View Product/i });
    await user.click(viewBtn);

    expect(api.recordFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        client_id: 'demo_ecommerce',
        user_id: '9272',
        item_id: 'prod_fb_target',
        action: 'CLICK',
        recommendation_id: 'rec_fb_link_005',
      })
    );
  });

  // Test H: API failure renders a controlled user-facing error
  it('H: API failure renders controlled error without breaking UI', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockRejectedValueOnce(
      new Error('Inference feature contract violation: candidate items are missing required feature column feat_x')
    );

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Inference Error/i)).toBeInTheDocument();
      expect(screen.getByText(/Inference feature contract violation/i)).toBeInTheDocument();
    });
  });

  // Test J: No-active-model 503 state renders correctly
  it('J: No-active-model 503 error renders actionable user message', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockRejectedValueOnce(
      new Error('No active model is available for client demo_ecommerce. Train and activate a model version first.')
    );

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Inference Error/i)).toBeInTheDocument();
      expect(screen.getByText(/No active model is available for client demo_ecommerce/i)).toBeInTheDocument();
    });
  });

  // Test K: Client switching clears previous recommendation state
  it('K: Client switching clears previous recommendation state', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_switch_01',
      client_id: 'demo_ecommerce',
      user_id: '9272',
      status: 'warm_start',
      history_count: 5,
      candidate_count: 899,
      latency_ms: 18.2,
      modality_weights: { behavior: 0.6, content: 0.2, context: 0.2 },
      recommendations: [
        { rank: 1, item_id: '101', score: 0.88, metadata: {}, explanation: 'Warm pattern' }
      ]
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Product 101/i)).toBeInTheDocument();
    });

    // Switch client dropdown to movielens_dataset
    const clientSelect = screen.getByDisplayValue(/Demo E-Commerce/i);
    await user.selectOptions(clientSelect, 'movielens_dataset');

    // Previous recommendation result must be cleared immediately
    await waitFor(() => {
      expect(screen.queryByText(/Product 101/i)).not.toBeInTheDocument();
    });
  });

  // Test L: Generic MovieLens schema displays movie item labels and genre tags
  it('L: Generic MovieLens schema displays movie item labels and genre tags', async () => {
    const user = userEvent.setup();
    localStorage.setItem('recommender_client_id', 'movielens_dataset');

    api.getSchema.mockImplementation((cid) => {
      if (cid === 'movielens_dataset') {
        return Promise.resolve({
          column_mappings: {
            user_id: 'USER_ID',
            movie_id: 'ITEM_ID',
            interaction_datetime: 'TIMESTAMP',
            target: 'TARGET',
            genre_action: 'CONTENT_FEATURE',
          }
        });
      }
      return Promise.resolve({ column_mappings: {} });
    });

    api.getRecommendations.mockResolvedValueOnce({
      recommendation_id: 'rec_ml_01',
      client_id: 'movielens_dataset',
      user_id: '1',
      status: 'warm_start',
      history_count: 232,
      candidate_count: 6233,
      latency_ms: 25.4,
      modality_weights: { behavior: 0.85, content: 0.02, context: 0.13 },
      recommendations: [
        {
          rank: 1,
          item_id: '5288',
          movie_name: 'Action Movie Title',
          score: 0.9794,
          metadata: { movie_id: 5288, genre_action: 1 },
          explanation: 'Personalized from previous historical user interaction patterns.'
        }
      ]
    });

    renderWithProviders(<RecommendationPage />);
    await switchToProductMode(user);

    // Wait for MovieLens schema to finish loading
    await waitFor(() => {
      expect(screen.getByText(/No context features mapped for this client schema/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText('Action Movie Title')).toBeInTheDocument();
      expect(screen.getByText('ID: 5288')).toBeInTheDocument();
      expect(screen.getByText(/^action$/i)).toBeInTheDocument();
    });
  });

  // Test M: Recommendation mode toggle switches between Products and Visualizations
  it('M: Allows toggling between Product Recommendations and Visualization Recommendations', async () => {
    const user = userEvent.setup();
    renderWithProviders(<RecommendationPage />);

    // Initially in MLVis Visual Analytics mode (Default Primary)
    expect(screen.getByRole('button', { name: /Visual Analytics Recommender \(MLVis\)/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Product Recommender \(Two-Tower\)/i })).toBeInTheDocument();

    // Visual Analytics workspace is active by default
    await waitFor(() => {
      expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();
      expect(screen.queryByText(/Customer \/ User Identifier/i)).not.toBeInTheDocument();
    });

    // Switch to Product Recommender
    await user.click(screen.getByRole('button', { name: /Product Recommender \(Two-Tower\)/i }));

    // Product inputs should now be visible
    await waitFor(() => {
      expect(screen.getByText(/Customer \/ User Identifier/i)).toBeInTheDocument();
    });

    // Switch back to MLVis Visualization Recommender
    await user.click(screen.getByRole('button', { name: /Visual Analytics Recommender \(MLVis\)/i }));

    await waitFor(() => {
      expect(screen.queryByText(/Customer \/ User Identifier/i)).not.toBeInTheDocument();
      expect(screen.getByText(/Upload CSV/i)).toBeInTheDocument();
    });
  });
});
