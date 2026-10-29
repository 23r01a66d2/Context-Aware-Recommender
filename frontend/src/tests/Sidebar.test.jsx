import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import { ClientProvider } from '../context/ClientContext';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn(),
    getClients: vi.fn(),
  },
}));

function renderSidebar() {
  return render(
    <BrowserRouter>
      <ClientProvider>
        <Sidebar />
      </ClientProvider>
    </BrowserRouter>
  );
}

describe('Sidebar Component UI Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    api.getHealth.mockResolvedValue({ status: 'ok' });
    api.getClients.mockResolvedValue([
      {
        client_id: 'demo_ecommerce',
        name: 'Indian E-Commerce Benchmark',
        active_model_version: '1',
      },
    ]);
  });

  it('renders "Smart Recommendations" title and "Personalized suggestions across domains" subtitle', () => {
    renderSidebar();

    // Check heading
    expect(screen.getByText('Smart Recommendations')).toBeInTheDocument();

    // Check subtitle
    expect(
      screen.getByText('Personalized suggestions across domains')
    ).toBeInTheDocument();

    // Verify "AI-Powered" is completely removed
    expect(screen.queryByText('AI-Powered')).toBeNull();
    expect(
      screen.queryByText(/Better Recommendations Across Domains/i)
    ).toBeNull();
  });

  it('keeps the callout card styling and classes for responsiveness', () => {
    const { container } = renderSidebar();

    const titleEl = screen.getByText('Smart Recommendations');
    expect(titleEl).toHaveClass('truncate');

    const cardContainer = titleEl.closest('.rounded-xl');
    expect(cardContainer).toBeInTheDocument();
    expect(cardContainer).toHaveClass('flex', 'items-center', 'gap-3');
  });
});
