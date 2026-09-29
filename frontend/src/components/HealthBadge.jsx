import React from 'react';
import { useClient } from '../context/ClientContext';
import { Activity, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function HealthBadge() {
  const { health } = useClient();

  if (health.loading) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs bg-slate-800 text-slate-400 border border-slate-700">
        <Activity className="w-3.5 h-3.5 animate-spin" />
        <span>Checking backend...</span>
      </div>
    );
  }

  if (health.healthy) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/80" title={health.data?.version ? `Backend API v${health.data.version}` : 'FastAPI Backend Online'}>
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span>
        <CheckCircle2 className="w-3.5 h-3.5" />
        <span>{health.data?.version ? `Backend Online (v${health.data.version})` : 'Backend Online'}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-rose-950/80 text-rose-400 border border-rose-800/80" title={health.error || 'Connection Failed'}>
      <AlertCircle className="w-3.5 h-3.5" />
      <span>Backend Offline</span>
    </div>
  );
}
