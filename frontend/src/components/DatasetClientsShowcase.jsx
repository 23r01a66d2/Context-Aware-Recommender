import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../context/ClientContext';
import { api } from '../services/api';
import {
  Film,
  ShoppingBag,
  Compass,
  Database,
  Users,
  Box,
  Activity,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  ChevronRight,
  Cpu,
  RefreshCw,
  Loader2,
  Check,
  ArrowRight,
  Play
} from 'lucide-react';

const DIAGNOSTIC_TEST_CLIENTS = new Set([
  'contract_violation_client',
  'client_no_model',
  'client_without_active_model',
  'temp_live_del_1',
]);

const isDiagnosticClient = (id) => {
  if (!id) return false;
  return DIAGNOSTIC_TEST_CLIENTS.has(id);
};

export default function DatasetClientsShowcase({ onClientSelected, className = '', showSwitchModel = true }) {
  const { clients, selectedClientId, selectClient, refreshClients } = useClient();
  const [overviews, setOverviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionMessage, setActionMessage] = useState(null);

  const fetchOverviews = async () => {
    setLoading(true);
    try {
      if (api.getClientsOverview) {
        const data = await api.getClientsOverview();
        if (Array.isArray(data) && data.length > 0) {
          const normalData = data.filter((c) => !isDiagnosticClient(c.client_id));
          setOverviews(normalData);
          return;
        }
      }
      // Fallback from clients list if overview not available
      const fallback = (clients || [])
        .filter((c) => !isDiagnosticClient(c.client_id))
        .map((c) => ({
          client_id: c.client_id,
          name: c.name || c.client_id,
          description: c.description || '',
          domain: c.client_id === 'demo_ecommerce' ? 'Products / E-Commerce' : c.client_id === 'movielens_dataset' ? 'Movies & Entertainment' : c.client_id === 'fashion_client' ? 'Fashion & Apparel' : 'Custom Domain',
          dataset_name: c.client_id === 'demo_ecommerce' ? 'Indian E-Commerce Customer Behavior' : c.client_id === 'movielens_dataset' ? 'MovieLens 100k' : 'Custom Dataset',
          user_label: c.client_id === 'demo_ecommerce' ? 'Customers' : 'Users',
          item_label: c.client_id === 'demo_ecommerce' ? 'Products' : c.client_id === 'movielens_dataset' ? 'Movies' : 'Items',
          users_count: c.client_id === 'demo_ecommerce' ? 8442 : c.client_id === 'movielens_dataset' ? 610 : c.client_id === 'fashion_client' ? 520 : 100,
          items_count: c.client_id === 'demo_ecommerce' ? 899 : c.client_id === 'movielens_dataset' ? 9724 : c.client_id === 'fashion_client' ? 8934 : 50,
          interactions_count: c.client_id === 'demo_ecommerce' ? 25000 : c.client_id === 'movielens_dataset' ? 100836 : c.client_id === 'fashion_client' ? 22760 : 1000,
          interaction_types: ['Impressions', 'Views', 'Clicks', 'Ignores'],
          available_models: c.active_model_version ? [c.active_model_version] : ['v1'],
          active_model_version: c.active_model_version || 'v1',
          supports_context: true,
          context_features: ['session_context'],
          supports_implicit_feedback: true,
          is_system: !!c.is_system,
          is_inference_ready: c.is_inference_ready !== false,
          readiness_category: c.readiness_category || 'Ready for Inference',
        }));
      setOverviews(fallback);
    } catch (err) {
      console.error('Failed to load client overviews:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviews();
  }, [clients]);

  const handleSelectClient = (clientId) => {
    selectClient(clientId);
    if (onClientSelected) {
      onClientSelected(clientId);
    }
    setActionMessage(`Switched active domain to ${clientId}`);
    setTimeout(() => setActionMessage(null), 3000);
  };

  const getDomainIcon = (clientId, domain) => {
    const dLower = (domain || '').toLowerCase();
    const cLower = (clientId || '').toLowerCase();
    if (cLower.includes('movie') || dLower.includes('movie') || dLower.includes('film')) {
      return <Film className="w-4 h-4 text-amber-500" />;
    }
    if (cLower.includes('ecom') || dLower.includes('product') || dLower.includes('commerce') || dLower.includes('retail') || cLower.includes('fashion')) {
      return <ShoppingBag className="w-4 h-4 text-teal-500" />;
    }
    if (cLower.includes('ar') || dLower.includes('augmented') || dLower.includes('tour')) {
      return <Compass className="w-4 h-4 text-sky-500" />;
    }
    return <Database className="w-4 h-4 text-purple-500" />;
  };

  const [activeTab, setActiveTab] = useState('all');

  const displayOverviews = overviews.filter((item) => {
    if (activeTab === 'all') return true;
    const cat = item.readiness_category || (item.is_inference_ready ? 'Ready for Inference' : 'Ready for Training');
    if (activeTab === 'inference') return cat === 'Ready for Inference';
    if (activeTab === 'training') return cat === 'Ready for Training';
    if (activeTab === 'schema') return cat === 'Needs Schema Mapping';
    return true;
  });

  const counts = {
    all: overviews.length,
    inference: overviews.filter(o => o.readiness_category === 'Ready for Inference' || (o.is_inference_ready && !o.readiness_category)).length,
    training: overviews.filter(o => o.readiness_category === 'Ready for Training').length,
    schema: overviews.filter(o => o.readiness_category === 'Needs Schema Mapping').length,
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header with Title and Filter Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 mb-1">
            <Sparkles className="w-3 h-3" /> Multi-Domain Recommendation Catalog
          </div>
          <h2 className="text-base sm:text-lg font-bold dark:text-coolwhite text-slate-900 tracking-tight flex items-center gap-2">
            Available Datasets & Recommendation Clients
          </h2>
          <p className="text-xs dark:text-muted text-slate-500">
            Active recommendation clients and deployed neural model checkpoints.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {actionMessage && (
            <span className="text-xs text-teal-600 dark:text-teal-300 bg-teal-500/10 border border-teal-500/30 px-2.5 py-1 rounded-md flex items-center gap-1 animate-fadeIn">
              <Check className="w-3.5 h-3.5 text-teal-500" /> {actionMessage}
            </span>
          )}
          <Link
            to="/admin/clients"
            className="text-xs font-semibold text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1 shrink-0"
          >
            View All <ArrowRight className="w-3.5 h-3.5" />
          </Link>
          <button
            onClick={fetchOverviews}
            disabled={loading}
            className="p-1.5 rounded-lg dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900 transition-colors text-xs flex items-center gap-1 cursor-pointer shadow-sm"
            title="Refresh Datasets"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Category Filter Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs border-b dark:border-navy-800 border-slate-200">
        <button
          onClick={() => setActiveTab('all')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 shrink-0 cursor-pointer ${
            activeTab === 'all'
              ? 'dark:bg-teal-500/15 bg-teal-50 text-teal-600 dark:text-teal-300 border dark:border-teal-500/30 border-teal-200 font-bold'
              : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
          }`}
        >
          <span>All ({counts.all})</span>
        </button>
        <button
          onClick={() => setActiveTab('inference')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 shrink-0 cursor-pointer ${
            activeTab === 'inference'
              ? 'dark:bg-teal-500/15 bg-teal-50 text-teal-600 dark:text-teal-300 border dark:border-teal-500/30 border-teal-200 font-bold'
              : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
          }`}
        >
          <CheckCircle2 className="w-3.5 h-3.5 text-teal-500" />
          <span>Inference Ready ({counts.inference})</span>
        </button>
        <button
          onClick={() => setActiveTab('training')}
          className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 shrink-0 cursor-pointer ${
            activeTab === 'training'
              ? 'dark:bg-sky-500/15 bg-sky-50 text-sky-600 dark:text-sky-300 border dark:border-sky-500/30 border-sky-200 font-bold'
              : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
          }`}
        >
          <Cpu className="w-3.5 h-3.5 text-sky-500" />
          <span>Ready for Training ({counts.training})</span>
        </button>
        {counts.schema > 0 && (
          <button
            onClick={() => setActiveTab('schema')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5 shrink-0 cursor-pointer ${
              activeTab === 'schema'
                ? 'dark:bg-amber-500/15 bg-amber-50 text-amber-600 dark:text-amber-300 border dark:border-amber-500/30 border-amber-200 font-bold'
                : 'dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900'
            }`}
          >
            <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
            <span>Needs Mapping ({counts.schema})</span>
          </button>
        )}
      </div>

      {/* Grid of Clean Client Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {loading && displayOverviews.length === 0 ? (
          <div className="col-span-full py-10 text-center dark:text-muted text-slate-500 text-xs flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin text-teal-500" />
            Loading recommendation clients...
          </div>
        ) : (
          displayOverviews.map((client) => {
            const isSelected = selectedClientId === client.client_id;
            const isReady = client.is_inference_ready !== false;

            return (
              <div
                key={client.client_id}
                onClick={() => handleSelectClient(client.client_id)}
                className={`rounded-2xl p-4 cursor-pointer transition-all border flex flex-col justify-between group shadow-sm ${
                  isSelected
                    ? 'dark:bg-navy-850 bg-teal-50/40 dark:border-teal-500/60 border-teal-500 ring-2 ring-teal-500/20'
                    : 'dark:bg-navy-900 bg-white dark:border-navy-800 border-slate-200 dark:hover:bg-navy-850 hover:bg-slate-50 hover:border-slate-300 dark:hover:border-navy-700'
                }`}
              >
                <div className="space-y-3">
                  {/* Top: Domain Icon + Readable Name & Readiness Badge */}
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="p-2 rounded-xl dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 shrink-0">
                        {getDomainIcon(client.client_id, client.domain)}
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-sm font-bold dark:text-coolwhite text-slate-900 truncate group-hover:text-teal-600 dark:group-hover:text-teal-300 transition-colors">
                          {client.name}
                        </h3>
                        <p className="text-[11px] dark:text-muted text-slate-500 truncate font-mono">
                          {client.client_id}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-bold border shrink-0 ${
                        isReady
                          ? 'dark:bg-teal-500/15 bg-teal-50 text-teal-600 dark:text-teal-300 dark:border-teal-500/30 border-teal-200'
                          : 'dark:bg-amber-500/15 bg-amber-50 text-amber-600 dark:text-amber-300 dark:border-amber-500/30 border-amber-200'
                      }`}
                    >
                      {isReady ? 'Inference Ready' : 'Training Ready'}
                    </span>
                  </div>

                  {/* Summary Metrics */}
                  <div className="grid grid-cols-3 gap-1 p-2 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-center text-xs">
                    <div>
                      <div className="text-[10px] dark:text-muted text-slate-500">Users</div>
                      <div className="text-xs font-bold dark:text-coolwhite text-slate-900 mt-0.5 font-mono">
                        {Number(client.users_count || 0).toLocaleString()}
                      </div>
                    </div>
                    <div className="border-x dark:border-navy-800 border-slate-200">
                      <div className="text-[10px] dark:text-muted text-slate-500">Items</div>
                      <div className="text-xs font-bold dark:text-coolwhite text-slate-900 mt-0.5 font-mono">
                        {Number(client.items_count || 0).toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] dark:text-muted text-slate-500">Events</div>
                      <div className="text-xs font-bold dark:text-coolwhite text-slate-900 mt-0.5 font-mono">
                        {Number(client.interactions_count || 0).toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Footer: Active Model tag & Clear Action Button */}
                <div className="pt-3 mt-3 border-t dark:border-navy-800 border-slate-200 flex items-center justify-between text-xs">
                  <div>
                    <div className="text-[10px] dark:text-muted text-slate-500">Active Model</div>
                    <div className="text-xs font-bold text-teal-600 dark:text-teal-400 font-mono">
                      {client.active_model_version ? `v${client.active_model_version}` : 'v1'}
                    </div>
                  </div>

                  <Link
                    to="/recommend"
                    onClick={(e) => {
                      e.stopPropagation();
                      selectClient(client.client_id);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-teal-500/10 dark:bg-teal-500/20 text-teal-700 dark:text-teal-300 font-semibold text-[11px] hover:bg-teal-500 hover:text-slate-950 transition-all border border-teal-500/30"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    Launch Demo
                  </Link>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
