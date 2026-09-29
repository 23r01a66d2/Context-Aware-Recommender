import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import { Building2, Plus, ArrowRight, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

export default function ClientsPage() {
  const { clients, selectedClientId, selectClient, refreshClients, loadingClients } = useClient();
  const [refreshing, setRefreshing] = useState(false);

  const handleManualRefresh = async () => {
    setRefreshing(true);
    await refreshClients();
    setRefreshing(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Building2 className="w-6 h-6 text-emerald-400" />
            Client Organizations
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Manage multi-tenant isolated workspaces, schema definitions, and model versions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleManualRefresh}
            disabled={refreshing}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Client List"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          </button>
          <Link
            to="/admin/clients/new"
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-sm transition-all"
          >
            <Plus className="w-4 h-4" />
            Add New Client
          </Link>
        </div>
      </div>

      {/* Clients Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
              <tr>
                <th className="py-3 px-4">Client ID</th>
                <th className="py-3 px-4">Organization Name</th>
                <th className="py-3 px-4">Active Model</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {loadingClients ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-500">
                    Loading clients...
                  </td>
                </tr>
              ) : clients.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-500">
                    No clients found. Click "Add New Client" to create one.
                  </td>
                </tr>
              ) : (
                clients.map((client) => {
                  const isSelected = client.client_id === selectedClientId;
                  return (
                    <tr
                      key={client.client_id}
                      className={`hover:bg-slate-800/40 transition-colors ${
                        isSelected ? 'bg-emerald-500/[0.04]' : ''
                      }`}
                    >
                      <td className="py-3.5 px-4 font-mono font-medium text-slate-200">
                        {client.client_id}
                        {isSelected && (
                          <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-sans border border-emerald-500/20">
                            Active Scope
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 font-medium text-white">{client.name || '—'}</td>
                      <td className="py-3.5 px-4">
                        {client.active_model_version ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            <CheckCircle2 className="w-3 h-3" />
                            {client.active_model_version}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-slate-800 text-slate-400 border border-slate-700">
                            <AlertCircle className="w-3 h-3" />
                            No Active Model
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="text-slate-400">Ready</span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {!isSelected && (
                            <button
                              onClick={() => selectClient(client.client_id)}
                              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-[11px] transition-colors"
                            >
                              Switch Scope
                            </button>
                          )}
                          <Link
                            to={`/admin/clients/${client.client_id}/dataset`}
                            onClick={() => selectClient(client.client_id)}
                            className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-emerald-400 transition-colors"
                            title="Manage Dataset & Models"
                          >
                            <ArrowRight className="w-4 h-4" />
                          </Link>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
