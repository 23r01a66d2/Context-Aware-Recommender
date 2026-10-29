import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useClient } from '../context/ClientContext';
import HealthBadge from './HealthBadge';
import ThemeSwitcher from './ThemeSwitcher';
import {
  Layers,
  Sparkles,
  SlidersHorizontal,
  Plus,
  Bell,
  User,
  Settings
} from 'lucide-react';

export default function Navbar() {
  const { clients, selectedClientId, selectClient, loadingClients } = useClient();
  const location = useLocation();

  const isLiveDemo = location.pathname.startsWith('/recommend');

  return (
    <header className="sticky top-0 z-40 dark:bg-navy-950/95 bg-white/95 backdrop-blur-md border-b dark:border-navy-800 border-slate-200 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Logo / Brand */}
        <div className="flex items-center gap-3">
          <Link to="/admin" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-lg bg-teal-500 flex items-center justify-center shadow-md shadow-teal-500/20 group-hover:scale-105 transition-transform">
              <Layers className="w-5 h-5 text-navy-950 font-bold" />
            </div>
            <div>
              <div className="text-sm font-bold tracking-tight dark:text-coolwhite text-slate-900 flex items-center gap-1.5">
                Context-Aware Recommender
                <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20">
                  Platform
                </span>
              </div>
              <div className="text-[11px] dark:text-muted text-slate-500">
                Multi-Modal Recommendation Platform
              </div>
            </div>
          </Link>
        </div>

        {/* Center / Navigation Links */}
        <div className="hidden md:flex items-center gap-1 dark:bg-navy-900/90 bg-slate-100 p-1 rounded-lg border dark:border-navy-800 border-slate-200">
          <Link
            to="/admin"
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
              !isLiveDemo
                ? 'dark:bg-navy-800 bg-white dark:text-coolwhite text-slate-900 shadow-sm'
                : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            Admin Console
          </Link>
          <Link
            to="/recommend"
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
              isLiveDemo
                ? 'bg-teal-500/15 text-teal-700 dark:text-teal-300 border border-teal-500/30 shadow-sm font-semibold'
                : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-teal-500" />
            Live Recommendations
          </Link>
        </div>

        {/* Right actions: Theme Switcher + Client Switcher + Health + User */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Theme Switcher Component */}
          <ThemeSwitcher />

          {/* Client Switcher */}
          <div className="hidden lg:flex items-center gap-1.5 dark:bg-navy-900 bg-slate-100 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1">
            <span className="text-xs dark:text-muted text-slate-500 font-medium">Client:</span>
            {loadingClients ? (
              <span className="text-xs dark:text-muted text-slate-500">Loading...</span>
            ) : (
              <select
                value={selectedClientId}
                onChange={(e) => selectClient(e.target.value)}
                className="bg-transparent text-xs font-semibold text-teal-600 dark:text-teal-400 focus:outline-none cursor-pointer pr-1"
              >
                {clients.map((c) => (
                  <option key={c.client_id} value={c.client_id} className="dark:bg-navy-900 bg-white dark:text-coolwhite text-slate-900">
                    {c.name || c.client_id} {c.active_model_version ? `(${c.active_model_version})` : '(No Model)'}
                  </option>
                ))}
              </select>
            )}
            <Link
              to="/admin/clients/new"
              className="p-1 rounded dark:text-muted text-slate-500 hover:text-teal-500 hover:dark:bg-navy-800 hover:bg-slate-200 transition-colors ml-0.5"
              title="Add New Client"
            >
              <Plus className="w-3.5 h-3.5" />
            </Link>
          </div>

          {/* Live Backend Telemetry Health Badge */}
          <HealthBadge />

          {/* User Profile avatar as in reference */}
          <div className="flex items-center gap-2 pl-1 border-l dark:border-navy-800 border-slate-200">
            <div className="w-8 h-8 rounded-full bg-teal-500/20 text-teal-600 dark:text-teal-400 border border-teal-500/30 flex items-center justify-center font-bold text-xs">
              <User className="w-4 h-4" />
            </div>
            <div className="hidden xl:block text-left">
              <div className="text-xs font-bold dark:text-coolwhite text-slate-900 leading-tight">Admin</div>
              <div className="text-[10px] dark:text-muted text-slate-500 leading-tight">Administrator</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
