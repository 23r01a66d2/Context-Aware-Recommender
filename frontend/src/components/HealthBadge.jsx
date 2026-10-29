import React from 'react';
import { useClient } from '../context/ClientContext';
import { Activity, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function HealthBadge() {
  const { health } = useClient();

  if (health.loading) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs dark:bg-navy-900 bg-slate-100 dark:text-muted text-slate-500 border dark:border-navy-800 border-slate-200">
        <Activity className="w-3.5 h-3.5 animate-spin text-teal-500" />
        <span className="hidden sm:inline">Checking backend...</span>
      </div>
    );
  }

  if (health.healthy) {
    return (
      <div
        className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium dark:bg-teal-500/10 bg-teal-50 dark:text-teal-300 text-teal-700 border dark:border-teal-500/30 border-teal-200 shadow-sm"
        title={health.data?.version ? `Backend API v${health.data.version}` : 'FastAPI Backend Online'}
      >
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-500"></span>
        </span>
        <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" />
        <span>{health.data?.version ? `Online (v${health.data.version})` : 'Online'}</span>
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-300 dark:border-rose-800/80"
      title={health.error || 'Connection Failed'}
    >
      <AlertCircle className="w-3.5 h-3.5 text-rose-500" />
      <span>Offline</span>
    </div>
  );
}
