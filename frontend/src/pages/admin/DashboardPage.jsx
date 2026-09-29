import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import {
  Users,
  Box,
  FileSpreadsheet,
  Activity,
  ArrowRight,
  Sparkles,
  Cpu,
  Layers,
  ShieldCheck,
  BarChart2,
} from 'lucide-react';

export default function DashboardPage() {
  const { clients, selectedClientId, selectedClient, health } = useClient();
  const [stats, setStats] = useState({
    totalClients: 0,
    activeModels: 0,
    totalFeedback: 0,
    datasetStatus: null,
    modelCount: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboardData() {
      setLoading(true);
      try {
        const [feedbackData, modelsData, datasetData] = await Promise.allSettled([
          api.getFeedbackLog(selectedClientId),
          api.getModels(selectedClientId),
          api.getDataset(selectedClientId),
        ]);

        const totalClients = clients.length;
        const activeModels = clients.filter((c) => c.active_model_version).length;
        const totalFeedback = feedbackData.status === 'fulfilled' ? feedbackData.value.length : 0;
        const modelCount = modelsData.status === 'fulfilled' ? modelsData.value.length : 0;
        const datasetStatus = datasetData.status === 'fulfilled' ? datasetData.value : null;

        setStats({
          totalClients,
          activeModels,
          totalFeedback,
          datasetStatus,
          modelCount,
        });
      } catch (err) {
        console.error('Error loading dashboard stats:', err);
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, [selectedClientId, clients]);

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 relative overflow-hidden shadow-xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold mb-4 border border-emerald-500/20">
            <Sparkles className="w-3.5 h-3.5" /> Multi-Tenant Multi-Modal Neural Architecture
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Multi-Client Recommendation Platform
          </h1>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">
            Real-time candidate ranking powered by Two-Tower Neural Multi-Modal fusion, dynamic schema-driven feature transformations, strict chronological data leakage prevention, and zero-history cold-start adaptive gating.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Link
              to="/recommend"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs shadow-lg shadow-emerald-500/20 transition-all"
            >
              <Sparkles className="w-4 h-4" />
              Launch Live Recommender
            </Link>
            <Link
              to={`/admin/clients/${selectedClientId}/training`}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition-all"
            >
              <Cpu className="w-4 h-4" />
              Train Active Client ({selectedClientId})
            </Link>
          </div>
        </div>
      </div>

      {/* Overview Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Registered Clients</span>
            <Users className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black text-white mt-2">
            {loading ? '—' : stats.totalClients}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Multi-tenant isolated workspaces</div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Active Deployed Models</span>
            <Box className="w-4 h-4 text-teal-400" />
          </div>
          <div className="text-2xl font-black text-white mt-2">
            {loading ? '—' : stats.activeModels}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            {selectedClient?.active_model_version
              ? `Client '${selectedClientId}' on v${selectedClient.active_model_version}`
              : 'Selected client has no active model'}
          </div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Client Model Versions</span>
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-black text-white mt-2">
            {loading ? '—' : stats.modelCount}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Checkpoints in registry for {selectedClientId}</div>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Recorded Feedback Events</span>
            <Activity className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-black text-white mt-2">
            {loading ? '—' : stats.totalFeedback}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Linked to recommendation IDs</div>
        </div>
      </div>

      {/* Active Client Status & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Active Client Information Card */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Active Client Scope: {selectedClient?.name || selectedClientId}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Client ID: <span className="font-mono text-slate-300">{selectedClientId}</span>
              </p>
            </div>
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">
              {selectedClient?.active_model_version ? `Active Model: ${selectedClient.active_model_version}` : 'Untrained'}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium">Dataset Status:</span>
              <div className="font-semibold text-slate-200 mt-1">
                {stats.datasetStatus?.row_count
                  ? `${stats.datasetStatus.row_count.toLocaleString()} rows (${stats.datasetStatus.column_count} columns)`
                  : 'Dataset registered'}
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                Temporal Capability: {stats.datasetStatus?.has_timestamp !== false ? 'Enabled (Zero-leakage)' : 'Disabled'}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
              <span className="text-slate-400 font-medium">Inference Architecture:</span>
              <div className="font-semibold text-slate-200 mt-1">Two-Tower Multi-Modal</div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                Behavior + Content + Context + Cold-Start Gate
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between text-xs">
            <span className="text-slate-400">Need to configure another client?</span>
            <Link to="/admin/clients" className="text-emerald-400 hover:text-emerald-300 font-semibold flex items-center gap-1">
              Manage All Clients <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>

        {/* Right: Quick Action Links */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 space-y-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
            Pipeline Steps
          </h3>

          <Link
            to={`/admin/clients/${selectedClientId}/dataset`}
            className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
              <div>
                <div className="text-xs font-semibold text-white">1. Ingest Dataset</div>
                <div className="text-[11px] text-slate-400">Upload & inspect raw files</div>
              </div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
          </Link>

          <Link
            to={`/admin/clients/${selectedClientId}/schema`}
            className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Layers className="w-4 h-4 text-teal-400" />
              <div>
                <div className="text-xs font-semibold text-white">2. Schema Mapping</div>
                <div className="text-[11px] text-slate-400">Define user, item, feature roles</div>
              </div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
          </Link>

          <Link
            to={`/admin/clients/${selectedClientId}/training`}
            className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <div>
                <div className="text-xs font-semibold text-white">3. Train Model</div>
                <div className="text-[11px] text-slate-400">Live epoch progress & loss</div>
              </div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
          </Link>

          <Link
            to={`/admin/clients/${selectedClientId}/analytics`}
            className="flex items-center justify-between p-3 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 transition-colors group"
          >
            <div className="flex items-center gap-3">
              <BarChart2 className="w-4 h-4 text-amber-400" />
              <div>
                <div className="text-xs font-semibold text-white">4. Offline Analytics</div>
                <div className="text-[11px] text-slate-400">Precision, NDCG, Cold vs Warm</div>
              </div>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
          </Link>
        </div>
      </div>
    </div>
  );
}
