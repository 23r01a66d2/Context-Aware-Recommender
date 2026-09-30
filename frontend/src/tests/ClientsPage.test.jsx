import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import ClientsPage from '../pages/admin/ClientsPage';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
    deleteClient: vi.fn(),
  },
}));

function renderWithProviders(ui) {
  return render(
    <BrowserRouter>
      <ClientProvider>{ui}</ClientProvider>
    </BrowserRouter>
  );
}

describe('ClientsPage Client Deletion Tests', () => {
  const mockClients = [
    {
      client_id: 'demo_ecommerce',
      name: 'Indian E-Commerce Benchmark',
      active_model_version: 'v1',
      is_system: true,
    },
    {
      client_id: 'custom_generic_client',
      name: 'Custom Generic Organization',
      active_model_version: 'v1',
      is_system: false,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    api.getHealth.mockResolvedValue({ status: 'ok' });
    api.getClients.mockResolvedValue(mockClients);
  });

  it('Renders client list with protected reference badge for system client and delete button for generic client', async () => {
    renderWithProviders(<ClientsPage />);

    await waitFor(() => {
      expect(screen.getByText('demo_ecommerce')).toBeInTheDocument();
      expect(screen.getByText('custom_generic_client')).toBeInTheDocument();
    });

    expect(screen.getByText(/Protected Reference/i)).toBeInTheDocument();

    const deleteBtn = screen.getByTitle('Delete custom_generic_client');
    expect(deleteBtn).toBeInTheDocument();
    expect(deleteBtn).not.toBeDisabled();
  });

  it('Clicking delete on generic client opens confirmation modal with impact list and disabled submit', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ClientsPage />);

    await waitFor(() => {
      expect(screen.getByText('custom_generic_client')).toBeInTheDocument();
    });

    const deleteBtn = screen.getByTitle('Delete custom_generic_client');
    await user.click(deleteBtn);

    expect(screen.getByText('Delete Client Organization')).toBeInTheDocument();
    expect(screen.getByText(/This action permanently removes the client registration/i)).toBeInTheDocument();
    expect(screen.getByText(/Platform client registration/i)).toBeInTheDocument();
    expect(screen.getByText(/Model version registry entries & training history/i)).toBeInTheDocument();

    // Confirm button should be disabled before typing confirmation
    const submitBtn = screen.getByRole('button', { name: /Delete Client/i });
    expect(submitBtn).toBeDisabled();
  });

  it('Enables Delete button only when exact client_id is entered and submits deletion', async () => {
    const user = userEvent.setup();
    api.deleteClient.mockResolvedValueOnce({
      client_id: 'custom_generic_client',
      message: 'Client custom_generic_client deleted successfully',
      deleted_records: { datasets: 1, schema_mappings: 1 },
      physical_files_deleted: true,
    });

    renderWithProviders(<ClientsPage />);

    await waitFor(() => {
      expect(screen.getByText('custom_generic_client')).toBeInTheDocument();
    });

    await user.click(screen.getByTitle('Delete custom_generic_client'));

    const input = screen.getByPlaceholderText('custom_generic_client');
    const submitBtn = screen.getByRole('button', { name: /Delete Client/i });

    // Partial typing does not enable
    await user.type(input, 'custom_generic');
    expect(submitBtn).toBeDisabled();

    // Complete exact match enables
    await user.type(input, '_client');
    expect(submitBtn).not.toBeDisabled();

    // Toggle physical files checkbox
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    expect(checkbox).toBeChecked();

    // Click Delete
    await user.click(submitBtn);

    await waitFor(() => {
      expect(api.deleteClient).toHaveBeenCalledWith('custom_generic_client', true);
    });
  });

  it('Displays error message inside modal if deletion API fails', async () => {
    const user = userEvent.setup();
    api.deleteClient.mockRejectedValueOnce(new Error('Cannot delete client while a training run is active'));

    renderWithProviders(<ClientsPage />);

    await waitFor(() => {
      expect(screen.getByText('custom_generic_client')).toBeInTheDocument();
    });

    await user.click(screen.getByTitle('Delete custom_generic_client'));
    const input = screen.getByPlaceholderText('custom_generic_client');
    await user.type(input, 'custom_generic_client');

    const submitBtn = screen.getByRole('button', { name: /Delete Client/i });
    await user.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText(/Cannot delete client while a training run is active/i)).toBeInTheDocument();
    });
  });
});
