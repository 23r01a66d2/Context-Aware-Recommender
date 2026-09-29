import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import TrainingPage from '../pages/admin/TrainingPage';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    getTrainingStatus: vi.fn(),
    triggerTraining: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ClientProvider>{ui}</ClientProvider>
    </BrowserRouter>
  );
}

describe('TrainingPage Frontend Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getHealth.mockResolvedValue({ status: 'ok', service: 'context-aware-recommender' });
    api.getClients.mockResolvedValue([
      { client_id: 'demo_ecommerce', name: 'Demo E-Commerce', active_model_version: 'v1' },
    ]);
  });

  // Test I: Training status renders actual current_epoch / total_epochs without fabricated progress
  it('I: Training status renders actual current_epoch / total_epochs and losses', async () => {
    api.getTrainingStatus.mockResolvedValueOnce({
      client_id: 'demo_ecommerce',
      run_id: 'run_test_77',
      status: 'TRAINING',
      current_epoch: 7,
      total_epochs: 15,
      metrics: {
        train_loss: 0.3421,
        val_loss: 0.3892,
      },
      error_message: null,
      created_at: '2026-09-28T20:00:00Z',
      completed_at: null,
    });

    renderWithProviders(<TrainingPage />);

    await waitFor(() => {
      // Must display exact real epoch progress: (7 / 15)
      expect(screen.getByText(/Training Epochs \(7 \/ 15\)/i)).toBeInTheDocument();
    });

    // Real percentage: (7/15) * 100 = 46.66% -> 47%
    expect(screen.getByText('47%')).toBeInTheDocument();

    // Must display real un-fabricated loss metrics
    expect(screen.getByText(/Train Loss: 0.3421/i)).toBeInTheDocument();
    expect(screen.getByText(/Val Loss: 0.3892/i)).toBeInTheDocument();
  });

  // Test M: Documented default learning rate 0.001 is valid and does not trigger stepMismatch
  it('M: Documented default learning rate 0.001 is valid in the form', async () => {
    api.getTrainingStatus.mockResolvedValueOnce({
      client_id: 'demo_ecommerce',
      run_id: 'none',
      status: 'NOT_STARTED',
      current_epoch: 0,
      total_epochs: 15,
      metrics: null,
      error_message: null,
      created_at: null,
      completed_at: null,
    });

    renderWithProviders(<TrainingPage />);

    await waitFor(() => {
      expect(screen.getByText(/Start Training Run/i)).toBeInTheDocument();
    });

    const lrInput = screen.getByDisplayValue('0.001');
    expect(lrInput).toBeInTheDocument();
    expect(lrInput.checkValidity()).toBe(true);
    expect(lrInput.validity.stepMismatch).toBe(false);
  });
});
