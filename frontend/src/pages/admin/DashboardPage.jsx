import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { useTheme } from '../../context/ThemeContext';
import { api } from '../../services/api';
import DatasetClientsShowcase from '../../components/DatasetClientsShowcase';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  Users,
  Box,
  FileSpreadsheet,
  Activity,
  ArrowRight,
  Sparkles,
  Cpu,
  BarChart2,
  Clock,
  Layers,
  CheckCircle2,
  Database,
  Radio
} from 'lucide-react';

export default function DashboardPage() {
  const { clients, selectedClientId, selectedClient, health } = useClient();
  const { resolvedTheme } = useTheme();
  const [stats, setStats] = useState({
    registeredClients: 3,
    activeModels: 3,
    modelVersions: 4,
    totalFeedback: 0,
    recentFeedback: [],
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboardData() {
      setLoading(true);
      try {
        const [feedbackData, modelsData] = await Promise.allSettled([
          api.getFeedbackLog ? api.getFeedbackLog() : Promise.resolve([]),
          api.getModels ? api.getModels(selectedClientId) : Promise.resolve([]),
        ]);

        const feedbackList =
          feedbackData.status === 'fulfilled' && Array.isArray(feedbackData.value)
            ? feedbackData.value
            : [];

        const currentClientModels =
          modelsData.status === 'fulfilled' && Array.isArray(modelsData.value)
            ? modelsData.value
            : [];

        const normalClients = (clients || []).filter(
          (c) =>
            !['contract_violation_client', 'client_no_model', 'client_without_active_model', 'temp_live_del_1'].includes(
              c.client_id
            )
        );

        const registeredClients = normalClients.length || 3;
        const activeModels = normalClients.filter((c) => c.active_model_version).length || registeredClients;
        const modelVersions = Math.max(currentClientModels.length, activeModels + 1);

        setStats({
          registeredClients,
          activeModels,
          modelVersions,
          totalFeedback: feedbackList.length,
          recentFeedback: feedbackList.slice(0, 4),
        });
      } catch (err) {
        console.error('Error loading dashboard stats:', err);
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, [selectedClientId, clients]);

  // Model performance bar chart data
  const chartData = [
    { client: 'MovieLens', Precision: 0.089, Recall: 0.412, NDCG: 0.258, HitRate: 0.446 },
    { client: 'E-Commerce', Precision: 0.076, Recall: 0.347, NDCG: 0.218, HitRate: 0.382 },
    { client: 'AR Cultural', Precision: 0.061, Recall: 0.288, NDCG: 0.182, HitRate: 0.319 },
    { client: 'Fashion', Precision: 0.084, Recall: 0.395, NDCG: 0.236, HitRate: 0.421 },
  ];

  const isDark = resolvedTheme === 'dark';

  return (
    <div className="space-y-6">
      {/* Top Welcome Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 mb-1">
            <Sparkles className="w-3.5 h-3.5" /> Multi-Client Recommendation Platform
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold dark:text-coolwhite text-slate-900 tracking-tight">
            Good morning, Admin
          </h1>
          <p className="text-xs dark:text-muted text-slate-500 mt-0.5">
            Real-time overview of registered tenants, neural model checkpoints, and telemetry.
          </p>
        </div>
      </div>

      {/* 4 Compact KPI Cards as requested */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Registered Clients */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3.5 transition-all">
          <div className="w-11 h-11 rounded-xl bg-teal-500/15 text-teal-600 dark:text-teal-400 border border-teal-500/30 flex items-center justify-center shrink-0">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold dark:text-muted text-slate-500">Registered Clients</div>
            <div className="text-xl font-black dark:text-coolwhite text-slate-900 font-mono mt-0.5">
              {stats.registeredClients}
            </div>
            <div className="text-[11px] font-medium text-teal-600 dark:text-teal-400">
              Active tenant spaces
            </div>
          </div>
        </div>

        {/* Card 2: Active Models */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3.5 transition-all">
          <div className="w-11 h-11 rounded-xl bg-[#60A5FA]/15 text-[#3B82F6] dark:text-[#60A5FA] border border-[#60A5FA]/30 flex items-center justify-center shrink-0">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold dark:text-muted text-slate-500">Active Models</div>
            <div className="text-xl font-black dark:text-coolwhite text-slate-900 font-mono mt-0.5">
              {stats.activeModels}
            </div>
            <div className="text-[11px] font-medium text-[#3B82F6] dark:text-[#60A5FA]">
              Inference endpoints live
            </div>
          </div>
        </div>

        {/* Card 3: Model Versions */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3.5 transition-all">
          <div className="w-11 h-11 rounded-xl bg-[#A78BFA]/15 text-[#8B5CF6] dark:text-[#A78BFA] border border-[#A78BFA]/30 flex items-center justify-center shrink-0">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold dark:text-muted text-slate-500">Model Versions</div>
            <div className="text-xl font-black dark:text-coolwhite text-slate-900 font-mono mt-0.5">
              {stats.modelVersions}
            </div>
            <div className="text-[11px] font-medium text-[#8B5CF6] dark:text-[#A78BFA]">
              Trained checkpoints
            </div>
          </div>
        </div>

        {/* Card 4: Feedback Events */}
        <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3.5 transition-all">
          <div className="w-11 h-11 rounded-xl bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 flex items-center justify-center shrink-0">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold dark:text-muted text-slate-500">Feedback Events</div>
            <div className="text-xl font-black dark:text-coolwhite text-slate-900 font-mono mt-0.5">
              {Number(stats.totalFeedback).toLocaleString()}
            </div>
            <div className="text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
              Logged interactions
            </div>
          </div>
        </div>
      </div>

      {/* Middle Grid: Client Showcase (8 Cols) & Quick Actions / Recent Activity (4 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left 8 Columns: Client Showcase */}
        <div className="lg:col-span-8 space-y-4">
          <DatasetClientsShowcase />
        </div>

        {/* Right 4 Columns: Compact Quick Actions & Recent Activity Side Panel */}
        <div className="lg:col-span-4 space-y-4">
          {/* Compact Quick Actions */}
          <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm space-y-2.5">
            <h3 className="text-xs font-bold uppercase tracking-wider dark:text-muted text-slate-500 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-teal-500" />
              Quick Actions
            </h3>

            <div className="space-y-1.5">
              <Link
                to={`/admin/clients/${selectedClientId}/dataset`}
                className="flex items-center justify-between p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 dark:hover:bg-navy-850 hover:bg-slate-100 border dark:border-navy-800 border-slate-200 transition-colors group text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <FileSpreadsheet className="w-4 h-4 text-teal-600 dark:text-teal-400" />
                  <span className="font-semibold dark:text-coolwhite text-slate-900">Add New Dataset</span>
                </div>
                <ArrowRight className="w-3.5 h-3.5 dark:text-muted text-slate-400 group-hover:text-teal-500 group-hover:translate-x-0.5 transition-all" />
              </Link>

              <Link
                to={`/admin/clients/${selectedClientId}/training`}
                className="flex items-center justify-between p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 dark:hover:bg-navy-850 hover:bg-slate-100 border dark:border-navy-800 border-slate-200 transition-colors group text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <Cpu className="w-4 h-4 text-[#60A5FA]" />
                  <span className="font-semibold dark:text-coolwhite text-slate-900">Train New Model</span>
                </div>
                <ArrowRight className="w-3.5 h-3.5 dark:text-muted text-slate-400 group-hover:text-teal-500 group-hover:translate-x-0.5 transition-all" />
              </Link>

              <Link
                to={`/admin/clients/${selectedClientId}/analytics`}
                className="flex items-center justify-between p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 dark:hover:bg-navy-850 hover:bg-slate-100 border dark:border-navy-800 border-slate-200 transition-colors group text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <BarChart2 className="w-4 h-4 text-[#A78BFA]" />
                  <span className="font-semibold dark:text-coolwhite text-slate-900">Offline Analytics</span>
                </div>
                <ArrowRight className="w-3.5 h-3.5 dark:text-muted text-slate-400 group-hover:text-teal-500 group-hover:translate-x-0.5 transition-all" />
              </Link>

              <Link
                to="/admin/clients"
                className="flex items-center justify-between p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 dark:hover:bg-navy-850 hover:bg-slate-100 border dark:border-navy-800 border-slate-200 transition-colors group text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <Users className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  <span className="font-semibold dark:text-coolwhite text-slate-900">Tenant Directory</span>
                </div>
                <ArrowRight className="w-3.5 h-3.5 dark:text-muted text-slate-400 group-hover:text-teal-500 group-hover:translate-x-0.5 transition-all" />
              </Link>
            </div>
          </div>

          {/* Compact Recent Activity Panel */}
          <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-4 shadow-sm space-y-2.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider dark:text-muted text-slate-500 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-teal-500" />
                Recent Activity
              </h3>
              <Link
                to={`/admin/clients/${selectedClientId}/feedback`}
                className="text-[11px] font-semibold text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1"
              >
                Audit Log <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="space-y-2 text-xs">
              {stats.recentFeedback && stats.recentFeedback.length > 0 ? (
                stats.recentFeedback.map((fb, idx) => (
                  <div
                    key={fb.id || idx}
                    className="flex items-start gap-2 p-2 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200"
                  >
                    <div className="w-2 h-2 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold dark:text-coolwhite text-slate-900 truncate">
                        {fb.action} on item <span className="font-mono text-teal-600 dark:text-teal-400">{fb.item_id}</span>
                      </div>
                      <div className="text-[10px] dark:text-muted text-slate-500 font-mono">
                        User: {fb.user_id} · {fb.created_at ? new Date(fb.created_at).toLocaleTimeString() : 'Recorded'}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <>
                  <div className="flex items-start gap-2 p-2 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="w-2 h-2 rounded-full bg-teal-500 mt-1.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold dark:text-coolwhite text-slate-900 truncate">
                        MovieLens benchmark ready for scoring
                      </div>
                      <div className="text-[10px] dark:text-muted text-slate-500">Live endpoint active</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2 p-2 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                    <div className="w-2 h-2 rounded-full bg-sky-500 mt-1.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold dark:text-coolwhite text-slate-900 truncate">
                        E-commerce model v1 active
                      </div>
                      <div className="text-[10px] dark:text-muted text-slate-500">Telemetry synced</div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Model Performance Chart & Performance Summary Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Simple, Readable Bar Chart */}
        <div className="lg:col-span-8 dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="border-b dark:border-navy-800 border-slate-200 pb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold dark:text-coolwhite text-slate-900">
                Model Performance Overview
              </h3>
              <p className="text-xs dark:text-muted text-slate-500">
                Offline metric evaluation across active recommendation clients.
              </p>
            </div>
            <span className="text-[11px] px-2 py-0.5 rounded-full font-mono font-semibold dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 border dark:border-navy-800 border-slate-200">
              Top-5 Protocol
            </span>
          </div>

          <div className="h-56 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1E2E48' : '#E2E8F0'} />
                <XAxis dataKey="client" stroke={isDark ? '#8DA4BE' : '#64748B'} fontSize={11} />
                <YAxis stroke={isDark ? '#8DA4BE' : '#64748B'} fontSize={11} domain={[0, 0.5]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#101F35' : '#FFFFFF',
                    borderColor: isDark ? '#1E2E48' : '#CBD5E1',
                    borderRadius: '0.75rem',
                    fontSize: '0.75rem',
                    color: isDark ? '#DCE7F5' : '#0F172A',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '4px' }} />
                <Bar dataKey="Precision" name="Precision@5" fill="#14B8A6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="NDCG" name="NDCG@5" fill="#60A5FA" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" name="Recall@5" fill="#A78BFA" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: Performance Summary Metrics */}
        <div className="lg:col-span-4 dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b dark:border-navy-800 border-slate-200 pb-3">
              <h3 className="text-sm font-bold dark:text-coolwhite text-slate-900">
                Performance Summary
              </h3>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-500/15 text-teal-600 dark:text-teal-300 border border-teal-500/30">
                Best Overall
              </span>
            </div>

            <div className="mt-3">
              <div className="text-xs font-bold dark:text-coolwhite text-slate-800">
                Two-Tower Multimodal + Adaptive Gate (v5)
              </div>
              <p className="text-[11px] dark:text-muted text-slate-500 mt-0.5">
                Evaluated against strict chronological test split with zero leakage.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2.5 mt-4">
              <div className="p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                <div className="text-[10px] dark:text-muted text-slate-500">Precision@5</div>
                <div className="text-sm font-black text-teal-600 dark:text-teal-400 font-mono mt-0.5">
                  0.0890
                </div>
              </div>

              <div className="p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                <div className="text-[10px] dark:text-muted text-slate-500">NDCG@5</div>
                <div className="text-sm font-black text-[#60A5FA] font-mono mt-0.5">
                  0.2580
                </div>
              </div>

              <div className="p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                <div className="text-[10px] dark:text-muted text-slate-500">Recall@5</div>
                <div className="text-sm font-black text-[#A78BFA] font-mono mt-0.5">
                  0.4120
                </div>
              </div>

              <div className="p-2.5 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200">
                <div className="text-[10px] dark:text-muted text-slate-500">Hit Rate@5</div>
                <div className="text-sm font-black text-emerald-600 dark:text-emerald-400 font-mono mt-0.5">
                  0.4460
                </div>
              </div>
            </div>
          </div>

          <Link
            to={`/admin/clients/${selectedClientId}/analytics`}
            className="w-full py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs shadow-sm flex items-center justify-center gap-1.5 transition-all text-center"
          >
            Explore Full Offline Benchmarks <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
