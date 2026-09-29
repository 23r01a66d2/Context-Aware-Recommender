import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import RecommendationPage from '../pages/user/RecommendationPage';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    getRecommendations: vi.fn(),
    recordFeedback: vi.fn(),
    getSchema: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ClientProvider>{ui}</ClientProvider>
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
  });

  // Test A: Recommendation page renders
  it('A: Recommendation page renders essential controls', async () => {
    renderWithProviders(<RecommendationPage />);

    expect(screen.getByText(/Live Candidate Scoring & Recommendation Demo/i)).toBeInTheDocument();
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
    const submitBtn = screen.getByRole('button', { name: /Score & Rank Candidates/i });
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/New User — Cold Start/i)).toBeInTheDocument();
    });

    // Verify modality weights rendered
    expect(screen.getByText(/Modality Attention Fusion Weights/i)).toBeInTheDocument();
    expect(screen.getByText('0.008')).toBeInTheDocument();
    expect(screen.getByText('0.542')).toBeInTheDocument();
    expect(screen.getByText('0.450')).toBeInTheDocument();
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

  // Test F: Feedback button sends recommendation_id, client_id, user_id, item_id, action
  it('F: Feedback button submits correct payload with recommendation_id linkage', async () => {
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

    api.recordFeedback.mockResolvedValueOnce({
      id: 101,
      recommendation_id: 'rec_fb_link_005',
      client_id: 'demo_ecommerce',
      user_id: 'user_fb_test',
      item_id: 'prod_fb_target',
      action: 'ACCEPT',
      timestamp: '2026-09-28T20:00:00Z',
    });

    renderWithProviders(<RecommendationPage />);
    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Product prod_fb_target/i)).toBeInTheDocument();
    });

    const acceptBtn = screen.getByRole('button', { name: /Accept/i });
    await user.click(acceptBtn);

    expect(api.recordFeedback).toHaveBeenCalledWith({
      client_id: 'demo_ecommerce',
      user_id: '9272',
      item_id: 'prod_fb_target',
      action: 'ACCEPT',
      recommendation_id: 'rec_fb_link_005',
    });
  });

  // Test H: API failure renders a controlled user-facing error
  it('H: API failure renders controlled error without breaking UI', async () => {
    const user = userEvent.setup();
    api.getRecommendations.mockRejectedValueOnce(
      new Error('Inference feature contract violation: candidate items are missing required feature column feat_x')
    );

    renderWithProviders(<RecommendationPage />);
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
          score: 0.9794,
          metadata: { movie_id: 5288, genre_action: 1 },
          explanation: 'Personalized from previous historical user interaction patterns.'
        }
      ]
    });

    renderWithProviders(<RecommendationPage />);

    // Wait for MovieLens schema to finish loading
    await waitFor(() => {
      expect(screen.getByText(/No context features mapped for this client schema/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Score & Rank Candidates/i }));

    await waitFor(() => {
      expect(screen.getByText(/Movie 5288/i)).toBeInTheDocument();
      expect(screen.getByText(/^action$/i)).toBeInTheDocument();
    });
  });
});
