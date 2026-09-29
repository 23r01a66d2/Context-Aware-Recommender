import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

const ClientContext = createContext(null);

function getStoredClientId() {
  try {
    return localStorage.getItem('recommender_client_id') || 'demo_ecommerce';
  } catch (_) {
    return 'demo_ecommerce';
  }
}

function setStoredClientId(id) {
  try {
    localStorage.setItem('recommender_client_id', id);
  } catch (_) {}
}

export function ClientProvider({ children }) {
  const [clients, setClients] = useState([]);
  const [selectedClientId, setSelectedClientId] = useState(getStoredClientId);
  const [loadingClients, setLoadingClients] = useState(true);
  const [health, setHealth] = useState({ healthy: false, loading: true, data: null, error: null });

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.getHealth();
      setHealth({ healthy: data.status === 'ok' || data.status === 'healthy', loading: false, data, error: null });
    } catch (err) {
      setHealth({ healthy: false, loading: false, data: null, error: err.message });
    }
  }, []);

  const fetchClients = useCallback(async () => {
    setLoadingClients(true);
    try {
      const data = await api.getClients();
      const clientList = Array.isArray(data) ? data : (data?.clients || []);
      setClients(clientList);
      if (clientList.length > 0) {
        setSelectedClientId((current) => {
          const exists = clientList.some(c => c.client_id === current);
          if (!exists) {
            const defaultClient = clientList.find(c => c.client_id === 'demo_ecommerce') || clientList[0];
            setStoredClientId(defaultClient.client_id);
            return defaultClient.client_id;
          }
          return current;
        });
      }
    } catch (err) {
      console.error('Failed to load clients:', err);
    } finally {
      setLoadingClients(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    fetchClients();
    const healthInterval = setInterval(fetchHealth, 15000);
    return () => clearInterval(healthInterval);
  }, [fetchHealth, fetchClients]);

  const selectClient = (clientId) => {
    setSelectedClientId(clientId);
    setStoredClientId(clientId);
  };

  const selectedClient = Array.isArray(clients) ? (clients.find(c => c.client_id === selectedClientId) || null) : null;

  return (
    <ClientContext.Provider
      value={{
        clients,
        selectedClientId,
        selectedClient,
        selectClient,
        loadingClients,
        refreshClients: fetchClients,
        health,
        refreshHealth: fetchHealth,
      }}
    >
      {children}
    </ClientContext.Provider>
  );
}

export function useClient() {
  const context = useContext(ClientContext);
  if (!context) {
    throw new Error('useClient must be used within a ClientProvider');
  }
  return context;
}
