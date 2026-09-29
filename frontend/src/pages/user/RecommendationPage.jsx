import React, { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import ModalityWeightBar from '../../components/ModalityWeightBar';
import {
  Sparkles,
  Zap,
  Snowflake,
  Flame,
  MousePointer,
  CheckCircle2,
  XCircle,
  ShoppingBag,
  Sliders,
  Filter,
  RefreshCw,
  AlertCircle,
  HelpCircle,
  ArrowRight,
} from 'lucide-react';

export default function RecommendationPage() {
  const { clients, selectedClientId, selectClient } = useClient();
  const isDemoEcommerce = selectedClientId === 'demo_ecommerce';

  // Inputs
  const [userId, setUserId] = useState('9272');
  const [topK, setTopK] = useState(5);

  // Demo E-Commerce Context Variables
  const [deviceType, setDeviceType] = useState('Mobile');
  const [channel, setChannel] = useState('Direct');
  const [season, setSeason] = useState('Autumn');
  const [location, setLocation] = useState(42);

  // Filters
  const [categoryFilter, setCategoryFilter] = useState('');
  const [maxPrice, setMaxPrice] = useState('');

  // Generic Client Schema & Context
  const [clientSchema, setClientSchema] = useState(null);
  const [genericContext, setGenericContext] = useState({});

  // Execution State
  const [loading, setLoading] = useState(false);
  const [recResponse, setRecResponse] = useState(null);
  const [error, setError] = useState(null);
  const [feedbackSuccess, setFeedbackSuccess] = useState({});

  useEffect(() => {
    let isMounted = true;
    // Clear previous client execution results, context values, and filters
    setRecResponse(null);
    setError(null);
    setCategoryFilter('');
    setMaxPrice('');
    setFeedbackSuccess({});
    setGenericContext({});

    if (selectedClientId === 'demo_ecommerce') {
      setUserId('9272');
      setClientSchema(null);
    } else if (selectedClientId === 'context_aware_ar_interaction') {
      setUserId('U153');
    } else if (selectedClientId === 'movielens_dataset') {
      setUserId('1');
    } else {
      setUserId('user_1');
    }

    if (api.getSchema) {
      api.getSchema(selectedClientId)
        .then((schema) => {
          if (!isMounted) return;
          setClientSchema(schema);
          const defaults = {};
          if (schema?.column_mappings) {
            Object.entries(schema.column_mappings).forEach(([col, role]) => {
              if (role === 'CONTEXT_FEATURE') {
                defaults[col] = '';
              }
            });
          }
          setGenericContext(defaults);
        })
        .catch(() => {
          if (isMounted) setClientSchema(null);
        });
    }

    return () => {
      isMounted = false;
    };
  }, [selectedClientId]);

  // Schema-driven dynamic terminology (neutral defaults, client-appropriate overrides)
  const userLabel = useMemo(() => {
    if (clientSchema?.column_mappings) {
      for (const [col, role] of Object.entries(clientSchema.column_mappings)) {
        if (role === 'USER_ID') {
          if (col.toLowerCase().includes('customer')) return 'Customer';
          if (col.toLowerCase().includes('user')) return 'User';
          return col.replace(/_id$/i, '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        }
      }
    }
    if (isDemoEcommerce) return 'Customer';
    return 'User';
  }, [clientSchema, isDemoEcommerce]);

  const itemLabel = useMemo(() => {
    if (clientSchema?.column_mappings) {
      for (const [col, role] of Object.entries(clientSchema.column_mappings)) {
        if (role === 'ITEM_ID') {
          if (col.toLowerCase().includes('movie')) return 'Movie';
          if (col.toLowerCase().includes('product')) return 'Product';
          if (col.toLowerCase().includes('loc')) return 'Location';
          if (col.toLowerCase().includes('item')) return 'Item';
          return col.replace(/_id$/i, '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        }
      }
    }
    if (isDemoEcommerce) return 'Product';
    return 'Item';
  }, [clientSchema, isDemoEcommerce]);

  const handleGenerate = async (overrideUserId) => {
    const activeUser = overrideUserId !== undefined ? overrideUserId : userId;
    setLoading(true);
    setError(null);

    try {
      let contextPayload = {};
      let filtersPayload = {};

      if (isDemoEcommerce) {
        contextPayload = {
          device_type: deviceType,
          marketing_channel: channel,
          visit_season: season,
          location: Number(location),
          visit_day: 15,
          visit_month: 10,
          visit_weekday: 3,
        };

        if (categoryFilter.trim()) {
          filtersPayload.category = categoryFilter.trim();
        }
        if (maxPrice && !isNaN(Number(maxPrice))) {
          filtersPayload.max_price = Number(maxPrice);
        }
      } else {
        // Send only non-empty context features configured in the client's schema
        Object.entries(genericContext).forEach(([k, v]) => {
          if (v !== undefined && v !== null && String(v).trim() !== '') {
            contextPayload[k] = v;
          }
        });
      }

      const res = await api.getRecommendations({
        client_id: selectedClientId,
        user_id: activeUser.trim(),
        top_k: Number(topK),
        context: contextPayload,
        filters: filtersPayload,
        include_cold_start_weights: true,
      });

      setRecResponse(res);
      setFeedbackSuccess({});
    } catch (err) {
      const msg = err.message || '';
      if (msg.includes('Failed to fetch') || err.name === 'TypeError') {
        setError('Network Error: Unable to connect to backend server at http://127.0.0.1:8000. Please ensure the server is running.');
      } else {
        setError(msg || 'Inference engine failed to generate recommendations.');
      }
      setRecResponse(null);
    } finally {
      setLoading(false);
    }
  };

  const handleTryColdStart = () => {
    let coldId;
    if (selectedClientId === 'context_aware_ar_interaction') {
      coldId = `NEW_COLD_USER_${Math.floor(1000 + Math.random() * 9000)}`;
    } else if (selectedClientId === 'movielens_dataset') {
      coldId = '99999';
    } else {
      coldId = `cold_user_${Math.floor(1000 + Math.random() * 9000)}`;
    }
    setUserId(coldId);
    handleGenerate(coldId);
  };

  const handleSendFeedback = async (itemId, action) => {
    if (!recResponse) return;
    try {
      await api.recordFeedback({
        client_id: selectedClientId,
        user_id: userId,
        item_id: String(itemId),
        action: action,
        recommendation_id: recResponse.recommendation_id,
      });

      setFeedbackSuccess((prev) => ({
        ...prev,
        [itemId]: action,
      }));
    } catch (err) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  };

  const isCold = recResponse?.status === 'cold_start';

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold mb-2 border border-emerald-500/20">
            <Sparkles className="w-3.5 h-3.5" /> Real-Time Online Inference Engine
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Live Candidate Scoring & Recommendation Demo
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Execute real-time candidate generation and scoring directly against the active PyTorch model checkpoint. Observe dynamic modality attention weighting and zero-history cold-start adaptation.
          </p>
        </div>

        {/* Client Switcher in Demo */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 flex items-center gap-3">
          <span className="text-xs text-slate-400 font-medium">Active Client:</span>
          <select
            value={selectedClientId}
            onChange={(e) => selectClient(e.target.value)}
            className="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-xs font-semibold text-emerald-400 focus:outline-none cursor-pointer"
          >
            {clients.map((c) => (
              <option key={c.client_id} value={c.client_id}>
                {c.name || c.client_id} {c.active_model_version ? `(${c.active_model_version})` : '(No Model)'}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Interactive Controls Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input Controls Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400" />
              1. User & Request Settings
            </h2>
          </div>

          {/* User ID with Cold-Start Button */}
          <div className="space-y-2">
            <label className="block text-xs font-semibold text-slate-300">
              {userLabel === 'Customer' ? 'Customer / User' : userLabel} Identifier
            </label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder={`e.g. ${isDemoEcommerce ? '9272' : selectedClientId === 'context_aware_ar_interaction' ? 'U153' : '1'} (warm ${userLabel.toLowerCase()})`}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-200 focus:border-emerald-500"
              />
              <button
                type="button"
                onClick={handleTryColdStart}
                className="shrink-0 px-3 py-2 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/30 text-xs font-semibold flex items-center gap-1.5 transition-colors"
                title={`Generates a zero-history new ${userLabel.toLowerCase()} ID`}
              >
                <Snowflake className="w-3.5 h-3.5" />
                Try New User (Cold)
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              Known {userLabel.toLowerCase()}s (e.g. <code className="text-slate-400">{isDemoEcommerce ? '9272' : selectedClientId === 'context_aware_ar_interaction' ? 'U153' : '1'}</code>) activate behavioral history. New {userLabel.toLowerCase()}s trigger cold-start gating.
            </p>
          </div>

          {/* Top-K Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Top-K Recommendations ({topK})
            </label>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              className="w-full accent-emerald-500 cursor-pointer"
            />
          </div>

          {/* Context Attributes */}
          <div className="space-y-3 pt-3 border-t border-slate-800/80">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Real-Time Context Attributes
              </h3>
              {!isDemoEcommerce && (
                <span className="text-[10px] text-emerald-400 font-mono">Schema-Driven</span>
              )}
            </div>

            {isDemoEcommerce ? (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Device Type</label>
                  <select
                    value={deviceType}
                    onChange={(e) => setDeviceType(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500"
                  >
                    <option value="Mobile">Mobile</option>
                    <option value="Desktop">Desktop</option>
                    <option value="Tablet">Tablet</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">Channel</label>
                  <select
                    value={channel}
                    onChange={(e) => setChannel(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500"
                  >
                    <option value="Direct">Direct</option>
                    <option value="Organic">Organic</option>
                    <option value="Social">Social</option>
                    <option value="Paid">Paid Search</option>
                    <option value="Email">Email</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">Season</label>
                  <select
                    value={season}
                    onChange={(e) => setSeason(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500"
                  >
                    <option value="Autumn">Autumn</option>
                    <option value="Winter">Winter</option>
                    <option value="Spring">Spring</option>
                    <option value="Summer">Summer</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">Location ({location})</label>
                  <input
                    type="number"
                    min="0"
                    max="225"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>
            ) : Object.keys(genericContext).length > 0 ? (
              <div className="grid grid-cols-2 gap-3 text-xs max-h-48 overflow-y-auto pr-1">
                {Object.keys(genericContext).map((col) => (
                  <div key={col}>
                    <label className="block text-slate-400 mb-1 capitalize truncate" title={col}>
                      {col.replace(/_/g, ' ')}
                    </label>
                    <input
                      type="text"
                      placeholder={`e.g. ${col}`}
                      value={genericContext[col] || ''}
                      onChange={(e) =>
                        setGenericContext((prev) => ({ ...prev, [col]: e.target.value }))
                      }
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500 font-mono text-[11px]"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[11px] text-slate-500 italic">
                No context features mapped for this client schema.
              </p>
            )}
          </div>

          {/* Catalog Filters */}
          <div className="space-y-3 pt-3 border-t border-slate-800/80">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-slate-400" />
              Candidate Catalog Filters
            </h3>

            {isDemoEcommerce ? (
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Category Filter</label>
                  <input
                    type="text"
                    placeholder="e.g. 1 or Electronics"
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500"
                  />
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">Max Price ($)</label>
                  <input
                    type="number"
                    placeholder="e.g. 500"
                    value={maxPrice}
                    onChange={(e) => setMaxPrice(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 focus:border-emerald-500 font-mono"
                  />
                </div>
              </div>
            ) : (
              <p className="text-[11px] text-slate-500 italic">
                All candidates in client catalog eligible for scoring (unfiltered).
              </p>
            )}
          </div>

          {/* Generate Button */}
          <button
            type="button"
            onClick={() => handleGenerate()}
            disabled={loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 disabled:opacity-50 text-slate-950 font-extrabold text-xs shadow-lg shadow-emerald-500/20 flex items-center justify-center gap-2 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Scoring Catalog Candidates...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 fill-slate-950" />
                Score & Rank Candidates
              </>
            )}
          </button>
        </div>

        {/* Right: Results & Real-Time Inspection */}
        <div className="lg:col-span-2 space-y-5">
          {error && (
            <div className="p-4 rounded-xl bg-rose-950/70 border border-rose-800/80 text-rose-300 text-xs flex items-start gap-2.5 shadow-sm">
              <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
              <div>
                <div className="font-bold">Inference Error</div>
                <div className="mt-0.5">{error}</div>
              </div>
            </div>
          )}

          {recResponse && (
            <div className="space-y-4">
              {/* Telemetry Header */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 shadow-sm">
                <div className="flex items-center gap-3">
                  {isCold ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-sky-500/10 text-sky-400 border border-sky-500/20">
                      <Snowflake className="w-3.5 h-3.5" />
                      New User — Cold Start
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                      <Flame className="w-3.5 h-3.5" />
                      Warm Start
                    </span>
                  )}
                  <span className="text-xs text-slate-400 font-mono">
                    ID: {recResponse.recommendation_id}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
                  <Zap className="w-3.5 h-3.5" />
                  Latency: {(recResponse.latency_ms ?? recResponse.inference_latency_ms)?.toFixed(1) || '0.0'} ms
                </div>
              </div>

              {/* Dynamic Modality Weights Bar */}
              <ModalityWeightBar
                weights={recResponse.modality_weights}
                isColdStart={isCold}
              />

              {/* Ranked Candidate List */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs text-slate-400 px-1">
                  <span>Top-{recResponse.recommendations?.length} Scored Candidates</span>
                  <span className="text-[11px] font-medium text-slate-500">
                    Candidate Pool Size: {recResponse.candidate_count ?? recResponse.candidate_pool_size ?? '—'} items
                  </span>
                </div>

                {recResponse.recommendations?.map((item) => {
                  const meta = item.item_metadata || item.metadata || item.attributes || {};
                  const lastFeedback = feedbackSuccess[item.item_id];

                  return (
                    <div
                      key={item.item_id}
                      className="bg-slate-900/90 border border-slate-800 hover:border-slate-700 rounded-xl p-4 shadow-sm transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
                    >
                      {/* Left: Rank & Item Info */}
                      <div className="flex items-start gap-3.5">
                        <div className="w-8 h-8 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center font-bold font-mono text-xs text-emerald-400 shrink-0">
                          #{item.rank}
                        </div>
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-bold text-white font-mono">
                              {itemLabel} {item.item_id}
                            </span>
                            {meta.product_category !== undefined && (
                              <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                                Cat: {meta.product_category}
                              </span>
                            )}
                            {meta.unit_price !== undefined && (
                              <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/40 font-mono">
                                ${meta.unit_price}
                              </span>
                            )}
                            {meta.cultural_interest_category !== undefined && (
                              <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-950/60 text-indigo-300 border border-indigo-800/40">
                                {meta.cultural_interest_category}
                              </span>
                            )}
                            {Object.entries(meta)
                              .filter(([k, v]) => k.startsWith('genre_') && (v === 1 || v === '1' || v === true))
                              .slice(0, 3)
                              .map(([k]) => (
                                <span key={k} className="text-[11px] px-2 py-0.5 rounded bg-purple-950/60 text-purple-300 border border-purple-800/40">
                                  {k.replace('genre_', '').replace(/_/g, ' ')}
                                </span>
                              ))}
                          </div>

                          {/* Grounded Explanation */}
                          <p className="text-xs text-slate-400 mt-1">
                            {item.explanation ||
                              (isCold
                                ? 'Prioritized via contextual affinity and catalog feature metadata.'
                                : `Personalized from previous historical ${userLabel.toLowerCase()} interaction patterns.`)}
                          </p>
                        </div>
                      </div>

                      {/* Right: Recommendation Score & Feedback Actions */}
                      <div className="flex flex-col sm:items-end gap-2 shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-slate-800">
                        {/* Recommendation Score Badge */}
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] text-slate-400 font-medium">
                            Recommendation Score:
                          </span>
                          <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-emerald-400 font-mono text-xs font-extrabold shadow-inner">
                            {item.score !== undefined ? item.score.toFixed(4) : '0.8420'}
                          </span>
                        </div>

                        {/* Interactive Feedback Buttons */}
                        <div className="flex items-center gap-1.5 text-xs">
                          <button
                            type="button"
                            onClick={() => handleSendFeedback(item.item_id, 'CLICK')}
                            className={`p-1.5 rounded transition-colors text-[11px] flex items-center gap-1 ${
                              lastFeedback === 'CLICK'
                                ? 'bg-sky-500 text-slate-950 font-bold'
                                : 'bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-sky-400 border border-slate-800'
                            }`}
                            title="Log CLICK feedback"
                          >
                            <MousePointer className="w-3 h-3" />
                            <span>Click</span>
                          </button>

                          <button
                            type="button"
                            onClick={() => handleSendFeedback(item.item_id, 'ACCEPT')}
                            className={`p-1.5 rounded transition-colors text-[11px] flex items-center gap-1 ${
                              lastFeedback === 'ACCEPT'
                                ? 'bg-emerald-500 text-slate-950 font-bold'
                                : 'bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-emerald-400 border border-slate-800'
                            }`}
                            title="Log ACCEPT feedback"
                          >
                            <CheckCircle2 className="w-3 h-3" />
                            <span>Accept</span>
                          </button>

                          <button
                            type="button"
                            onClick={() => handleSendFeedback(item.item_id, 'REJECT')}
                            className={`p-1.5 rounded transition-colors text-[11px] flex items-center gap-1 ${
                              lastFeedback === 'REJECT'
                                ? 'bg-rose-500 text-white font-bold'
                                : 'bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-rose-400 border border-slate-800'
                            }`}
                            title="Log REJECT feedback"
                          >
                            <XCircle className="w-3 h-3" />
                            <span>Reject</span>
                          </button>

                          {isDemoEcommerce && (
                            <button
                              type="button"
                              onClick={() => handleSendFeedback(item.item_id, 'PURCHASE')}
                              className={`p-1.5 rounded transition-colors text-[11px] flex items-center gap-1 ${
                                lastFeedback === 'PURCHASE'
                                  ? 'bg-purple-500 text-white font-bold'
                                  : 'bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-purple-400 border border-slate-800'
                              }`}
                              title="Log PURCHASE feedback"
                            >
                              <ShoppingBag className="w-3 h-3" />
                              <span>Purchase</span>
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {!recResponse && !error && (
            <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl p-12 text-center text-slate-500 text-xs space-y-3">
              <Sparkles className="w-8 h-8 text-slate-600 mx-auto" />
              <div className="font-semibold text-slate-400 text-sm">
                Ready for Real-Time Candidate Scoring
              </div>
              <p className="max-w-md mx-auto text-slate-500">
                Select context attributes on the left and click "Score & Rank Candidates" or test zero-history cold-start behavior.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
