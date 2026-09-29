import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { useClient } from '../context/ClientContext';
import {
  LayoutDashboard,
  Building2,
  FileSpreadsheet,
  Network,
  Cpu,
  Boxes,
  BarChart3,
  MessageSquare,
  Sparkles,
  ExternalLink,
} from 'lucide-react';

export default function Sidebar() {
  const { selectedClientId, selectedClient } = useClient();

  const navItems = [
    { name: 'Dashboard', path: '/admin', icon: LayoutDashboard, exact: true },
    { name: 'Clients', path: '/admin/clients', icon: Building2 },
    {
      name: 'Dataset & Ingestion',
      path: `/admin/clients/${selectedClientId}/dataset`,
      icon: FileSpreadsheet,
    },
    {
      name: 'Schema Mapping',
      path: `/admin/clients/${selectedClientId}/schema`,
      icon: Network,
    },
    {
      name: 'Model Training',
      path: `/admin/clients/${selectedClientId}/training`,
      icon: Cpu,
    },
    {
      name: 'Model Registry',
      path: `/admin/clients/${selectedClientId}/models`,
      icon: Boxes,
    },
    {
      name: 'Offline Analytics',
      path: `/admin/clients/${selectedClientId}/analytics`,
      icon: BarChart3,
    },
    {
      name: 'Feedback Audit Log',
      path: `/admin/clients/${selectedClientId}/feedback`,
      icon: MessageSquare,
    },
  ];

  return (
    <aside className="w-64 bg-slate-900/70 border-r border-slate-800 flex flex-col justify-between py-5 shrink-0 min-h-[calc(100vh-4rem)]">
      <div className="space-y-6 px-3">
        {/* Active Client Context Summary */}
        <div className="px-3 py-2.5 rounded-lg bg-slate-950/70 border border-slate-800">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Selected Scope
          </div>
          <div className="text-xs font-bold text-white truncate mt-0.5">
            {selectedClient?.name || selectedClientId}
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5 flex items-center justify-between">
            <span>ID: {selectedClientId}</span>
            <span className="text-emerald-400 font-medium">
              {selectedClient?.active_model_version ? `v:${selectedClient.active_model_version}` : 'untrained'}
            </span>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.name}
                to={item.path}
                end={item.exact}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Live Recommender Callout */}
      <div className="px-3 pt-4 border-t border-slate-800/80">
        <Link
          to="/recommend"
          className="flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-emerald-950/50 to-teal-950/50 border border-emerald-800/40 text-emerald-300 hover:border-emerald-600/60 transition-all group"
        >
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
            <div>
              <div className="text-xs font-bold text-white">Live Recommender</div>
              <div className="text-[10px] text-slate-400">Real-time candidate inference</div>
            </div>
          </div>
          <ExternalLink className="w-3.5 h-3.5 text-slate-500 group-hover:text-emerald-400 transition-colors" />
        </Link>
      </div>
    </aside>
  );
}
