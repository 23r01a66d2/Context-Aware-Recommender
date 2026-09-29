import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useClient } from '../context/ClientContext';
import HealthBadge from './HealthBadge';
import { Layers, Sparkles, SlidersHorizontal, Plus } from 'lucide-react';

export default function Navbar() {
  const { clients, selectedClientId, selectClient, loadingClients } = useClient();
  const location = useLocation();

  const isLiveDemo = location.pathname.startsWith('/recommend');

  return (
    <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Logo / Brand */}
        <div className="flex items-center gap-3">
          <Link to="/admin" className="flex items-center gap-2 group">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 group-hover:scale-105 transition-transform">
              <Layers className="w-5 h-5 text-slate-950 font-bold" />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
                Context-Aware Recommender
                <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  Platform
                </span>
              </div>
              <div className="text-[11px] text-slate-400">Multi-Modal Recommendation Platform</div>
            </div>
          </Link>
        </div>

        {/* Center / Navigation Links */}
        <div className="hidden md:flex items-center gap-1 bg-slate-950/60 p-1 rounded-lg border border-slate-800/80">
          <Link
            to="/admin"
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
              !isLiveDemo
                ? 'bg-slate-800 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            Admin Console
          </Link>
          <Link
            to="/recommend"
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
              isLiveDemo
                ? 'bg-gradient-to-r from-emerald-500/20 to-teal-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            Live Recommendations
          </Link>
        </div>

        {/* Right actions: Client Switcher + Health */}
        <div className="flex items-center gap-3">
          {/* Client Switcher */}
          <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1">
            <span className="text-xs text-slate-400 font-medium">Client:</span>
            {loadingClients ? (
              <span className="text-xs text-slate-500">Loading...</span>
            ) : (
              <select
                value={selectedClientId}
                onChange={(e) => selectClient(e.target.value)}
                className="bg-transparent text-xs font-semibold text-emerald-400 focus:outline-none cursor-pointer pr-1"
              >
                {clients.map((c) => (
                  <option key={c.client_id} value={c.client_id} className="bg-slate-900 text-slate-200">
                    {c.name || c.client_id} {c.active_model_version ? `(${c.active_model_version})` : '(No Model)'}
                  </option>
                ))}
              </select>
            )}
            <Link
              to="/admin/clients/new"
              className="p-1 rounded text-slate-400 hover:text-emerald-400 hover:bg-slate-800 transition-colors ml-1"
              title="Add New Client"
            >
              <Plus className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* Live Backend Telemetry Health Badge */}
          <HealthBadge />
        </div>
      </div>
    </header>
  );
}
