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
  LineChart,
  MessageSquare,
  Sparkles,
  Settings,
  ExternalLink,
  Shield,
  Layers
} from 'lucide-react';

export default function Sidebar() {
  const { selectedClientId, selectedClient } = useClient();

  const navItems = [
    { name: 'Home', path: '/admin', icon: LayoutDashboard, exact: true },
    {
      name: 'Models',
      path: '/admin/models',
      icon: Boxes,
    },
    {
      name: 'Visual Analytics (MLVis)',
      path: '/admin/models/mlvis/workspace?tab=visualizations',
      icon: LineChart,
    },
    {
      name: 'Recommendations',
      path: '/recommend',
      icon: Sparkles,
    },
    {
      name: 'Analytics',
      path: `/admin/clients/${selectedClientId}/analytics`,
      icon: BarChart3,
    },
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
      name: 'Feedback Audit Log',
      path: `/admin/clients/${selectedClientId}/feedback`,
      icon: MessageSquare,
    },
    {
      name: 'Settings',
      path: '/admin/settings',
      icon: Settings,
    },
  ];

  return (
    <aside className="w-64 dark:bg-navy-950/80 bg-white/80 border-r dark:border-navy-800 border-slate-200 flex flex-col justify-between py-5 shrink-0 min-h-[calc(100vh-4rem)] transition-colors duration-200">
      <div className="space-y-4 px-3">
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
                  `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? 'dark:bg-teal-500/15 bg-teal-50 text-teal-600 dark:text-teal-300 font-bold border dark:border-teal-500/30 border-teal-200 shadow-sm'
                      : 'dark:text-muted text-slate-600 dark:hover:text-coolwhite hover:text-slate-900 dark:hover:bg-navy-900/60 hover:bg-slate-100'
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

      {/* Bottom Smart Recommendations Callout */}
      <div className="px-3 pt-4 space-y-2">
        <div className="p-3.5 rounded-xl dark:bg-navy-900/90 bg-slate-100 border dark:border-navy-800 border-slate-200 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/30 flex items-center justify-center shrink-0">
            <Layers className="w-4 h-4 text-teal-500" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xs font-bold dark:text-coolwhite text-slate-900 truncate">Smart Recommendations</div>
            <div className="text-[10px] dark:text-muted text-slate-500 leading-tight">
              Personalized suggestions across domains
            </div>
          </div>
        </div>

        {/* Active Client Context Scope */}
        <div className="px-3 py-2 rounded-lg dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-[10px] dark:text-muted text-slate-500 flex items-center justify-between font-mono">
          <span className="truncate max-w-[120px]">{selectedClient?.name || selectedClientId}</span>
          <span className="text-teal-600 dark:text-teal-400 font-semibold">
            {selectedClient?.active_model_version ? `v${selectedClient.active_model_version}` : 'v1'}
          </span>
        </div>
      </div>
    </aside>
  );
}
