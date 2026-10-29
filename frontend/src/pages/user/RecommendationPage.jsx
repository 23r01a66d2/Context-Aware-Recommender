import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useClient } from '../../context/ClientContext';
import { api } from '../../services/api';
import ModalityWeightBar from '../../components/ModalityWeightBar';
import DatasetClientsShowcase from '../../components/DatasetClientsShowcase';
import VisualAnalyticsPage from '../admin/VisualAnalyticsPage';
import UserIdentityStep from '../../components/UserIdentityStep';
import {
  Sparkles,
  Zap,
  Snowflake,
  Flame,
  MousePointer,
  CheckCircle2,
  XCircle,
  ShoppingBag,
  ShoppingCart,
  Eye,
  EyeOff,
  Sliders,
  Filter,
  RefreshCw,
  AlertCircle,
  HelpCircle,
  ArrowRight,
  ExternalLink,
  Info,
  X,
  Cpu,
  Layers,
  Film,
  Check,
  UserCheck,
  Tag,
  ChevronDown,
  ChevronUp,
  LineChart
} from 'lucide-react';

export default function RecommendationPage() {
  const { clients, selectedClientId, selectedClient, selectClient, refreshClients } = useClient();
  const isDemoEcommerce = selectedClientId === 'demo_ecommerce';
  const isFashion = selectedClientId === 'fashion_client';

  // Recommendation Mode: 'visualizations' (Default Primary) | 'products' (Two-Tower)
  const [recommendationMode, setRecommendationMode] = useState('visualizations');

  // Inputs
  const [userId, setUserId] = useState('9272');
  const [topK, setTopK] = useState(5);

  // Models state for active client
  const [availableModels, setAvailableModels] = useState([]);
  const [activeModelTag, setActiveModelTag] = useState('');
  const [activatingModel, setActivatingModel] = useState(false);

  // Demo E-Commerce / Fashion Context Variables
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
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [behaviorNotification, setBehaviorNotification] = useState(null);
  const [showCatalogShowcase, setShowCatalogShowcase] = useState(false);

  // Tracking state refs for deduplication and dwell time
  const trackedImpressionsRef = useRef(new Set());
  const trackedViewsRef = useRef(new Set());
  const previousRecsRef = useRef(null);
  const interactedItemsRef = useRef(new Set());

  useEffect(() => {
    let isMounted = true;
    setRecResponse(null);
    setError(null);
    setCategoryFilter('');
    setMaxPrice('');
    setFeedbackSuccess({});
    setGenericContext({});
    setSelectedProduct(null);
    setBehaviorNotification(null);
    trackedImpressionsRef.current.clear();
    trackedViewsRef.current.clear();
    previousRecsRef.current = null;
    interactedItemsRef.current.clear();

    if (selectedClientId === 'demo_ecommerce') {
      setUserId('9272');
      setClientSchema(null);
    } else if (selectedClientId === 'context_aware_ar_interaction') {
      setUserId('U153');
    } else if (selectedClientId === 'movielens_dataset') {
      setUserId('1');
    } else if (selectedClientId === 'fashion_client') {
      setUserId('user_101');
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

    if (api.getModels) {
      api.getModels(selectedClientId)
        .then((list) => {
          if (!isMounted) return;
          const modelsList = Array.isArray(list) ? list : [];
          setAvailableModels(modelsList);
          const active = modelsList.find((m) => m.is_active);
          if (active) {
            setActiveModelTag(active.version_tag || active.version);
          } else if (modelsList.length > 0) {
            setActiveModelTag(modelsList[0].version_tag || modelsList[0].version);
          } else {
            setActiveModelTag('');
          }
        })
        .catch(() => {
          if (isMounted) setAvailableModels([]);
        });
    }

    return () => {
      isMounted = false;
    };
  }, [selectedClientId]);

  const handleSwitchModel = async (newVersion) => {
    if (!newVersion || newVersion === activeModelTag) return;
    setActivatingModel(true);
    try {
      if (api.activateModel) {
        await api.activateModel(selectedClientId, newVersion);
        setActiveModelTag(newVersion);
        if (api.getModels) {
          const list = await api.getModels(selectedClientId);
          setAvailableModels(list || []);
        }
        await refreshClients();
        showNotification(`Activated model version ${newVersion} for ${selectedClientId}.`);
      }
    } catch (err) {
      setError(`Failed to activate model version ${newVersion}: ${err.message}`);
    } finally {
      setActivatingModel(false);
    }
  };

  const safeRecordFeedback = (payload) => {
    try {
      const res = api.recordFeedback?.(payload);
      if (res && typeof res.catch === 'function') {
        res.catch(() => {});
      }
    } catch {
      // ignore
    }
  };

  // Automatic dwell time tracking (VIEW after 1.5s, LONG_VIEW after 4s)
  useEffect(() => {
    if (!recResponse || !recResponse.recommendations || recResponse.recommendations.length === 0) {
      return;
    }

    const recId = recResponse.recommendation_id;
    const currentItems = recResponse.recommendations;

    // 1. Log IMPRESSIONS once per recommendation event
    if (!trackedImpressionsRef.current.has(recId)) {
      trackedImpressionsRef.current.add(recId);
      currentItems.forEach((item) => {
        safeRecordFeedback({
          client_id: selectedClientId,
          user_id: userId.trim(),
          item_id: String(item.item_id),
          action: 'IMPRESSION',
          position: item.rank,
          recommendation_id: recId,
        });
      });
    }

    // 2. Timer for implicit VIEW (1.5 seconds)
    const viewTimer = setTimeout(() => {
      currentItems.slice(0, 3).forEach((item) => {
        const viewKey = `${recId}_${item.item_id}_VIEW`;
        if (!trackedViewsRef.current.has(viewKey)) {
          trackedViewsRef.current.add(viewKey);
          safeRecordFeedback({
            client_id: selectedClientId,
            user_id: userId.trim(),
            item_id: String(item.item_id),
            action: 'VIEW',
            position: item.rank,
            recommendation_id: recId,
          });
        }
      });
    }, 1500);

    // 3. Timer for implicit LONG_VIEW (4.0 seconds)
    const longViewTimer = setTimeout(() => {
      currentItems.slice(0, 2).forEach((item) => {
        const longViewKey = `${recId}_${item.item_id}_LONG_VIEW`;
        if (!trackedViewsRef.current.has(longViewKey)) {
          trackedViewsRef.current.add(longViewKey);
          safeRecordFeedback({
            client_id: selectedClientId,
            user_id: userId.trim(),
            item_id: String(item.item_id),
            action: 'LONG_VIEW',
            position: item.rank,
            recommendation_id: recId,
          });
        }
      });
    }, 4000);

    return () => {
      clearTimeout(viewTimer);
      clearTimeout(longViewTimer);
    };
  }, [recResponse, selectedClientId, userId]);

  // Schema-driven dynamic terminology
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
    if (isDemoEcommerce || isFashion) return 'Customer';
    return 'User';
  }, [clientSchema, isDemoEcommerce, isFashion]);

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
    if (isDemoEcommerce || isFashion) return 'Product';
    return 'Item';
  }, [clientSchema, isDemoEcommerce, isFashion]);

  const showNotification = (msg) => {
    setBehaviorNotification(msg);
    setTimeout(() => {
      setBehaviorNotification((prev) => (prev === msg ? null : prev));
    }, 3500);
  };

  const handleGenerate = async (overrideUserId) => {
    const activeUser = overrideUserId !== undefined ? overrideUserId : userId;
    setLoading(true);
    setError(null);
    setSelectedProduct(null);

    // Track IGNORE for un-interacted items from previous recommendation response
    if (previousRecsRef.current && previousRecsRef.current.recommendations) {
      const prevRecId = previousRecsRef.current.recommendation_id;
      const prevItems = previousRecsRef.current.recommendations;
      prevItems.forEach((it) => {
        if (!interactedItemsRef.current.has(String(it.item_id))) {
          safeRecordFeedback({
            client_id: selectedClientId,
            user_id: activeUser.trim(),
            item_id: String(it.item_id),
            action: 'IGNORE',
            position: it.rank,
            recommendation_id: prevRecId,
          });
        }
      });
    }

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
      } else if (isFashion) {
        contextPayload = {
          device_type: deviceType,
        };
        if (categoryFilter.trim()) {
          filtersPayload.category = categoryFilter.trim();
        }
        if (maxPrice && !isNaN(Number(maxPrice))) {
          filtersPayload.max_price = Number(maxPrice);
        }
      } else {
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
      previousRecsRef.current = res;
      interactedItemsRef.current.clear();
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
    } else if (selectedClientId === 'fashion_client') {
      coldId = `cold_fashion_${Math.floor(1000 + Math.random() * 9000)}`;
    } else {
      coldId = `cold_user_${Math.floor(1000 + Math.random() * 9000)}`;
    }
    setUserId(coldId);
    handleGenerate(coldId);
  };

  const handleNaturalInteraction = async (item, action = 'CLICK') => {
    if (!recResponse) return;
    const itemIdStr = String(item.item_id);
    interactedItemsRef.current.add(itemIdStr);

    try {
      await api.recordFeedback({
        client_id: selectedClientId,
        user_id: userId.trim(),
        item_id: itemIdStr,
        action: action,
        position: item.rank,
        recommendation_id: recResponse.recommendation_id,
      });

      setFeedbackSuccess((prev) => ({
        ...prev,
        [itemIdStr]: action,
      }));

      if (action === 'CLICK') {
        setSelectedProduct(item);
        showNotification(`Recorded CLICK on ${itemLabel} ${item.item_id} — Online reranker updated.`);
      } else if (action === 'PURCHASE') {
        showNotification(`Recorded PURCHASE on ${itemLabel} ${item.item_id} — Positive behavioral boost applied.`);
      } else if (action === 'ADD_TO_CART') {
        showNotification(`Recorded ADD_TO_CART for ${itemLabel} ${item.item_id} — Affinity score increased.`);
      } else {
        showNotification(`Recorded ${action} on ${itemLabel} ${item.item_id}`);
      }
    } catch (err) {
      console.warn('Feedback submission error:', err.message);
    }
  };

  const isCold = recResponse?.status === 'cold_start';

  return (
    <div className="space-y-6">
      {/* Compact Hero Banner */}
      <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-colors duration-200">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-teal-500/10 text-teal-600 dark:text-teal-400 text-xs font-semibold mb-1.5 border border-teal-500/20">
            <Sparkles className="w-3.5 h-3.5" /> Real-Time Online Inference Engine
          </div>
          <h1 className="text-xl sm:text-2xl font-bold dark:text-coolwhite text-slate-900 tracking-tight">
            Live Candidate Scoring & Recommendation Demo
          </h1>
          <p className="text-xs dark:text-muted text-slate-500 mt-0.5 max-w-2xl">
            Real-time candidate scoring with dynamic representation fusion across behavior, multimodal content, and context.
          </p>

          {/* Quick Domain Switcher Chips */}
          <div className="flex items-center gap-1.5 mt-3 flex-wrap">
            <span className="text-[11px] dark:text-muted text-slate-500 font-medium">Domain:</span>
            {clients.map((c) => {
              const isSelected = c.client_id === selectedClientId;
              return (
                <button
                  key={c.client_id}
                  onClick={() => selectClient(c.client_id)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5 ${
                    isSelected
                      ? 'dark:bg-teal-500/15 bg-teal-50 text-teal-600 dark:text-teal-300 border dark:border-teal-500/30 border-teal-200 font-bold'
                      : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 border dark:border-navy-800 border-slate-200 hover:dark:text-coolwhite hover:text-slate-900'
                  }`}
                >
                  {c.client_id === 'movielens_dataset' && <Film className="w-3 h-3 text-amber-500" />}
                  {(c.client_id === 'demo_ecommerce' || c.client_id === 'fashion_client') && <ShoppingBag className="w-3 h-3 text-teal-500" />}
                  <span>{c.name || c.client_id}</span>
                </button>
              );
            })}
            <button
              onClick={() => setShowCatalogShowcase(!showCatalogShowcase)}
              className="text-xs text-teal-600 dark:text-teal-400 hover:underline flex items-center gap-1 ml-1 cursor-pointer"
            >
              {showCatalogShowcase ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              <span>{showCatalogShowcase ? 'Hide Domains' : 'Browse All Domains'}</span>
            </button>
          </div>
        </div>

        {/* Selected Client Scope Info & Select Dropdown */}
        <div className="dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-2xl p-3.5 shrink-0 flex flex-col gap-1.5 min-w-[210px]">
          <div className="text-[10px] uppercase tracking-wider font-semibold dark:text-muted text-slate-500">Current Target Client</div>
          <select
            value={selectedClientId}
            onChange={(e) => selectClient(e.target.value)}
            className="dark:bg-navy-900 bg-white border dark:border-navy-700 border-slate-300 rounded-lg px-2 py-1 text-xs font-bold text-teal-600 dark:text-teal-400 focus:outline-none cursor-pointer"
          >
            {clients.map((c) => (
              <option key={c.client_id} value={c.client_id} className="dark:bg-navy-900 bg-white dark:text-coolwhite text-slate-900">
                {c.name || c.client_id} {c.active_model_version ? `(${c.active_model_version})` : '(No Model)'}
              </option>
            ))}
          </select>
          <div className="text-[10px] font-mono dark:text-muted text-slate-500">
            Active Version: <span className="text-teal-600 dark:text-teal-400 font-bold">{activeModelTag || selectedClient?.active_model_version || 'v1'}</span>
          </div>
        </div>
      </div>

      {/* Optional Expandable Domain Showcase */}
      {showCatalogShowcase && (
        <div className="animate-fadeIn">
          <DatasetClientsShowcase
            onClientSelected={() => setShowCatalogShowcase(false)}
          />
        </div>
      )}

      {/* Subtle Behavior Notification Bar */}
      {behaviorNotification && (
        <div className="p-3 rounded-2xl dark:bg-teal-500/10 bg-teal-50 border dark:border-teal-500/30 border-teal-200 text-teal-700 dark:text-teal-300 text-xs flex items-center justify-between shadow-sm animate-fadeIn">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-teal-500 fill-teal-500" />
            <span className="font-semibold">{behaviorNotification}</span>
          </div>
          <button
            onClick={() => setBehaviorNotification(null)}
            className="dark:text-muted text-slate-500 hover:dark:text-coolwhite hover:text-slate-900 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Recommendation Mode Selector Tabs */}
      <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-2.5 shadow-sm flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          {/* Option 1: Visual Analytics Recommender (MLVis) — Default & Primary */}
          <button
            type="button"
            onClick={() => setRecommendationMode('visualizations')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 cursor-pointer ${
              recommendationMode === 'visualizations'
                ? 'bg-teal-500 text-navy-950 shadow-sm'
                : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900 dark:border-navy-800 border-slate-200 border'
            }`}
          >
            <LineChart className="w-4 h-4" />
            <span>Visual Analytics Recommender (MLVis)</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded-full dark:bg-navy-900/40 bg-teal-600/20 text-current font-mono font-bold">
              Primary
            </span>
          </button>

          {/* Option 2: Product Recommender (Two-Tower) — Existing Model */}
          <button
            type="button"
            onClick={() => setRecommendationMode('products')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 cursor-pointer ${
              recommendationMode === 'products'
                ? 'bg-teal-500 text-navy-950 shadow-sm'
                : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 hover:dark:text-coolwhite hover:text-slate-900 dark:border-navy-800 border-slate-200 border'
            }`}
          >
            <ShoppingBag className="w-4 h-4" />
            <span>Product Recommender (Two-Tower)</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded-full dark:bg-navy-900/40 bg-teal-600/20 text-current font-mono">
              Two-Tower
            </span>
          </button>
        </div>

        <div className="text-xs dark:text-muted text-slate-500 flex items-center gap-1.5 px-2">
          <Info className="w-3.5 h-3.5 text-teal-500" />
          <span>
            {recommendationMode === 'products'
              ? 'Multi-modal behavioral and context scoring on product catalog'
              : 'Context-aware chart and visual encoding recommendations powered by MLVis'}
          </span>
        </div>
      </div>

      {/* Mode-Specific Content */}
      {recommendationMode === 'visualizations' ? (
        <div className="space-y-4">
          <VisualAnalyticsPage
            embeddedInModelWorkspace={true}
            lockedModel="mlvis"
            modelName="MLVis Visualization Recommender"
            hideHeader={true}
          />
        </div>
      ) : (
        <>
          {/* Main 2-Column Desktop Layout (5 cols Input, 7 cols Results) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: 5 Cols for Input Controls */}
        <div className="lg:col-span-5 space-y-4">
          <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-5 shadow-sm space-y-4 transition-colors duration-200">
            <div className="flex items-center justify-between border-b dark:border-navy-800 border-slate-200 pb-3">
              <h2 className="text-xs font-bold uppercase tracking-wider dark:text-muted text-slate-500 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-teal-500" />
                1. User & Request Settings
              </h2>
              <span className="text-[11px] font-mono text-teal-600 dark:text-teal-400 font-bold">
                {selectedClientId}
              </span>
            </div>

            {/* Model Switcher for Selected Client */}
            {availableModels.length > 0 && (
              <div className="p-3 rounded-xl dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="dark:text-muted text-slate-500 font-semibold flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-teal-500" />
                    Model Checkpoint:
                  </span>
                  {activatingModel && (
                    <span className="text-[10px] text-teal-600 dark:text-teal-400 font-medium animate-pulse">
                      Activating...
                    </span>
                  )}
                </div>
                <select
                  value={activeModelTag}
                  onChange={(e) => handleSwitchModel(e.target.value)}
                  disabled={activatingModel}
                  className="w-full dark:bg-navy-900 bg-white border dark:border-navy-700 border-slate-300 rounded-lg px-2.5 py-1.5 text-xs font-mono font-bold text-teal-600 dark:text-teal-400 focus:outline-none cursor-pointer"
                >
                  {availableModels.map((m) => {
                    const tag = m.version_tag || m.version;
                    return (
                      <option key={tag} value={tag} className="dark:bg-navy-900 bg-white dark:text-coolwhite text-slate-900">
                        {tag} {m.is_active ? '(Active Inference Target)' : ''}
                      </option>
                    );
                  })}
                </select>
              </div>
            )}

            {/* Real-time Cold Start / Warm Start Identification & Persistent History */}
            <UserIdentityStep
              clientId={selectedClientId}
              userId={userId}
              onUserIdChange={(newUid) => setUserId(newUid)}
              showHistoryDrawer={true}
            />

            {/* User ID with Cold-Start Button & Presets */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold dark:text-coolwhite text-slate-800">
                {userLabel === 'Customer' ? 'Customer / User' : userLabel} Identifier
              </label>

              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder={`e.g. ${isDemoEcommerce ? '9272' : isFashion ? 'user_101' : '1'}`}
                  className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-3 py-2 text-xs font-mono dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleTryColdStart}
                  className="shrink-0 px-3 py-2 rounded-lg dark:bg-[#38BDF8]/10 bg-sky-50 dark:text-[#38BDF8] text-sky-700 border dark:border-[#38BDF8]/30 border-sky-200 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                  title="Generates a zero-history new user ID"
                >
                  <Snowflake className="w-3.5 h-3.5" />
                  Try New User (Cold)
                </button>
              </div>

              {/* Quick Presets */}
              <div className="flex items-center gap-1.5 flex-wrap pt-1">
                <span className="text-[10px] dark:text-muted text-slate-500 font-medium">Presets:</span>
                {isDemoEcommerce && (
                  <>
                    <button
                      type="button"
                      onClick={() => { setUserId('9272'); handleGenerate('9272'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === '9272' ? 'dark:bg-teal-500/20 bg-teal-100 text-teal-700 dark:text-teal-300 border-teal-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 9272 (Warm - 8 visits)
                    </button>
                    <button
                      type="button"
                      onClick={() => { setUserId('7838'); handleGenerate('7838'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === '7838' ? 'dark:bg-amber-500/20 bg-amber-100 text-amber-700 dark:text-amber-300 border-amber-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 7838 (Cold - 1 visit)
                    </button>
                  </>
                )}
                {selectedClientId === 'movielens_dataset' && (
                  <>
                    <button
                      type="button"
                      onClick={() => { setUserId('1'); handleGenerate('1'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === '1' ? 'dark:bg-teal-500/20 bg-teal-100 text-teal-700 dark:text-teal-300 border-teal-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 1 (232 ratings)
                    </button>
                    <button
                      type="button"
                      onClick={() => { setUserId('2'); handleGenerate('2'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === '2' ? 'dark:bg-teal-500/20 bg-teal-100 text-teal-700 dark:text-teal-300 border-teal-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 2 (29 ratings)
                    </button>
                    <button
                      type="button"
                      onClick={() => { setUserId('99999'); handleGenerate('99999'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === '99999' ? 'dark:bg-sky-500/20 bg-sky-100 text-sky-700 dark:text-sky-300 border-sky-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 99999 (Cold)
                    </button>
                  </>
                )}
                {isFashion && (
                  <>
                    <button
                      type="button"
                      onClick={() => { setUserId('user_101'); handleGenerate('user_101'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === 'user_101' ? 'dark:bg-teal-500/20 bg-teal-100 text-teal-700 dark:text-teal-300 border-teal-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User 101 (Warm)
                    </button>
                    <button
                      type="button"
                      onClick={() => { setUserId('cold_user_1'); handleGenerate('cold_user_1'); }}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors cursor-pointer ${userId === 'cold_user_1' ? 'dark:bg-sky-500/20 bg-sky-100 text-sky-700 dark:text-sky-300 border-sky-400' : 'dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-600 dark:border-navy-800 border-slate-200 hover:text-slate-900'}`}
                    >
                      User Cold_1
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Top-K Selector */}
            <div>
              <div className="flex items-center justify-between text-xs font-semibold dark:text-coolwhite text-slate-800 mb-1">
                <span>Top-K Recommendations</span>
                <span className="font-mono text-teal-600 dark:text-teal-400 font-bold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                className="w-full accent-teal-500 cursor-pointer"
              />
            </div>

            {/* Dataset-Specific Context Attributes */}
            <div className="space-y-3 pt-3 border-t dark:border-navy-800 border-slate-200">
              <h3 className="text-xs font-bold uppercase tracking-wider dark:text-muted text-slate-500">
                Real-Time Context Attributes
              </h3>

              {isDemoEcommerce ? (
                <div className="grid grid-cols-2 gap-2.5 text-xs">
                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Device Type</label>
                    <select
                      value={deviceType}
                      onChange={(e) => setDeviceType(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none cursor-pointer"
                    >
                      <option value="Mobile">Mobile</option>
                      <option value="Desktop">Desktop</option>
                      <option value="Tablet">Tablet</option>
                    </select>
                  </div>

                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Marketing Channel</label>
                    <select
                      value={channel}
                      onChange={(e) => setChannel(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none cursor-pointer"
                    >
                      <option value="Direct">Direct</option>
                      <option value="Organic">Organic</option>
                      <option value="Social">Social</option>
                      <option value="Paid">Paid Search</option>
                      <option value="Email">Email</option>
                    </select>
                  </div>

                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Season</label>
                    <select
                      value={season}
                      onChange={(e) => setSeason(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none cursor-pointer"
                    >
                      <option value="Autumn">Autumn</option>
                      <option value="Winter">Winter</option>
                      <option value="Spring">Spring</option>
                      <option value="Summer">Summer</option>
                    </select>
                  </div>

                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Location ({location})</label>
                    <input
                      type="number"
                      min="0"
                      max="225"
                      value={location}
                      onChange={(e) => setLocation(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none font-mono"
                    />
                  </div>
                </div>
              ) : Object.keys(genericContext).length > 0 ? (
                <div className="grid grid-cols-2 gap-2.5 text-xs max-h-44 overflow-y-auto pr-1">
                  {Object.keys(genericContext).map((col) => (
                    <div key={col}>
                      <label className="block dark:text-muted text-slate-500 mb-1 capitalize truncate" title={col}>
                        {col.replace(/_/g, ' ')}
                      </label>
                      <input
                        type="text"
                        placeholder={`e.g. ${col}`}
                        value={genericContext[col] || ''}
                        onChange={(e) =>
                          setGenericContext((prev) => ({ ...prev, [col]: e.target.value }))
                        }
                        className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none font-mono text-[11px]"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-[11px] dark:text-muted text-slate-500 italic">
                  No context features mapped for this client schema.
                </p>
              )}
            </div>

            {/* Catalog Filters (Only shown if applicable) */}
            {(isDemoEcommerce || isFashion) && (
              <div className="space-y-3 pt-3 border-t dark:border-navy-800 border-slate-200">
                <h3 className="text-xs font-bold uppercase tracking-wider dark:text-muted text-slate-500 flex items-center gap-1.5">
                  <Filter className="w-3.5 h-3.5 dark:text-muted text-slate-500" />
                  Candidate Filters
                </h3>
                <div className="grid grid-cols-2 gap-2.5 text-xs">
                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Category Filter</label>
                    <input
                      type="text"
                      placeholder="e.g. Apparel"
                      value={categoryFilter}
                      onChange={(e) => setCategoryFilter(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block dark:text-muted text-slate-500 mb-1">Max Price ($)</label>
                    <input
                      type="number"
                      placeholder="e.g. 500"
                      value={maxPrice}
                      onChange={(e) => setMaxPrice(e.target.value)}
                      className="w-full dark:bg-navy-950 bg-slate-50 border dark:border-navy-800 border-slate-200 rounded-lg px-2.5 py-1.5 dark:text-coolwhite text-slate-900 focus:border-teal-500 focus:outline-none font-mono"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Generate Button */}
            <button
              type="button"
              onClick={() => handleGenerate()}
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-navy-950 font-bold text-xs shadow-sm flex items-center justify-center gap-2 transition-all cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Scoring Candidates...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 fill-navy-950" />
                  Score & Rank Candidates
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: 7 Cols for Recommendation Results */}
        <div className="lg:col-span-7 space-y-4">
          {error && (
            <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-300 dark:border-rose-800/80 text-rose-700 dark:text-rose-300 text-xs flex items-start gap-2.5 shadow-sm">
              <AlertCircle className="w-5 h-5 shrink-0 text-rose-500 mt-0.5" />
              <div>
                <div className="font-bold">Inference Error</div>
                <div className="mt-0.5">{error}</div>
              </div>
            </div>
          )}

          {recResponse && (
            <div className="space-y-4">
              {/* Telemetry Header */}
              <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-sm transition-colors duration-200">
                <div className="flex items-center gap-2.5">
                  {isCold ? (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold dark:bg-[#38BDF8]/10 bg-sky-50 dark:text-[#38BDF8] text-sky-700 border dark:border-[#38BDF8]/20 border-sky-200">
                      <Snowflake className="w-3 h-3" />
                      New User — Cold Start
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-500/20">
                      <Flame className="w-3 h-3" />
                      Warm Start
                    </span>
                  )}
                  <span className="text-xs dark:text-muted text-slate-500 font-mono">
                    ID: {recResponse.recommendation_id}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-xs font-mono text-teal-600 dark:text-teal-400 bg-teal-500/10 px-2.5 py-0.5 rounded-full border border-teal-500/20">
                  <Zap className="w-3 h-3" />
                  Latency: {(recResponse.latency_ms ?? recResponse.inference_latency_ms)?.toFixed(1) || '0.0'} ms
                </div>
              </div>

              {/* Dynamic Modality Weights Bar */}
              <ModalityWeightBar
                weights={recResponse.modality_weights}
                isColdStart={isCold}
              />

              {/* Ranked Candidate List */}
              <div className="space-y-2.5">
                <div className="flex items-center justify-between text-xs dark:text-muted text-slate-500 px-1">
                  <span className="font-semibold dark:text-coolwhite text-slate-800">
                    Top-{recResponse.recommendations?.length} Scored Candidates (Click item to interact)
                  </span>
                  <span className="text-[11px] dark:text-muted text-slate-500">
                    Candidate Pool Size: {recResponse.candidate_count ?? recResponse.candidate_pool_size ?? '—'} items
                  </span>
                </div>

                {recResponse.recommendations?.map((item) => {
                  const meta = item.item_metadata || item.metadata || item.attributes || {};
                  const lastAction = feedbackSuccess[item.item_id];

                  return (
                    <div
                      key={item.item_id}
                      onClick={() => handleNaturalInteraction(item, 'CLICK')}
                      className={`dark:bg-navy-900 bg-white border rounded-2xl p-3.5 shadow-sm transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3.5 cursor-pointer dark:hover:bg-navy-850 hover:bg-slate-50 ${
                        lastAction
                          ? 'border-teal-500 ring-1 ring-teal-500/20'
                          : 'dark:border-navy-800 border-slate-200 dark:hover:border-navy-700 hover:border-slate-300'
                      }`}
                    >
                      {/* Left: Rank, Thumbnail & Title */}
                      <div className="flex items-start gap-3 min-w-0">
                        <div className="w-7 h-7 rounded-lg dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 flex items-center justify-center font-bold font-mono text-xs text-teal-600 dark:text-teal-400 shrink-0">
                          #{item.rank}
                        </div>

                        {/* Image Thumbnail if present */}
                        {item.image_url && (
                          <div className="w-12 h-12 sm:w-14 sm:h-14 rounded-xl overflow-hidden dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 shrink-0 flex items-center justify-center">
                            <img
                              src={item.image_url}
                              alt={item.display_name || item.item_id}
                              className="w-full h-full object-cover"
                              onError={(e) => { e.target.style.display = 'none'; }}
                            />
                          </div>
                        )}

                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-xs sm:text-sm font-bold dark:text-coolwhite text-slate-900 flex items-center gap-1">
                              {item.display_name || item.movie_name || item.item_name || item.title || `${itemLabel} ${item.item_id}`}
                            </span>
                            <span className="text-[10px] px-1.5 py-0.2 rounded dark:bg-navy-950 bg-slate-100 dark:text-muted text-slate-500 font-mono border dark:border-navy-800 border-slate-200">
                              ID: {item.item_id}
                            </span>
                            {item.category && (
                              <span className="text-[10px] px-1.5 py-0.2 rounded dark:bg-navy-800 bg-slate-100 dark:text-coolwhite text-slate-800">
                                {item.category}
                              </span>
                            )}
                            {meta.product_category !== undefined && !item.category && (
                              <span className="text-[10px] px-1.5 py-0.2 rounded dark:bg-navy-800 bg-slate-100 dark:text-coolwhite text-slate-800">
                                {meta.product_category}
                              </span>
                            )}
                            {(item.price !== undefined && item.price !== null) ? (
                              <span className="text-[10px] px-1.5 py-0.2 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 font-mono font-bold">
                                ${item.price}
                              </span>
                            ) : meta.unit_price !== undefined ? (
                              <span className="text-[10px] px-1.5 py-0.2 rounded bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 font-mono font-bold">
                                ${meta.unit_price}
                              </span>
                            ) : null}
                            {Object.entries(meta)
                              .filter(([k, v]) => k.startsWith('genre_') && (v === 1 || v === '1' || v === true))
                              .slice(0, 3)
                              .map(([k]) => (
                                <span key={k} className="text-[10px] px-1.5 py-0.2 rounded dark:bg-purple-950 bg-purple-100 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                                  {k.replace('genre_', '').replace(/_/g, ' ')}
                                </span>
                              ))}
                          </div>

                          {/* Grounded Explanation */}
                          <p className="text-[11px] dark:text-muted text-slate-500 mt-1 line-clamp-1">
                            {item.explanation ||
                              (isCold
                                ? 'Prioritized via contextual affinity and catalog feature metadata.'
                                : `Personalized from previous historical ${userLabel.toLowerCase()} interaction patterns.`)}
                          </p>
                        </div>
                      </div>

                      {/* Right: Score & Actions */}
                      <div className="flex flex-col sm:items-end gap-1.5 shrink-0 border-t sm:border-t-0 pt-2 sm:pt-0 dark:border-navy-800 border-slate-200" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] dark:text-muted text-slate-500 font-medium">Recommendation Score:</span>
                          <span className="px-2 py-0.5 rounded-lg dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 text-teal-600 dark:text-teal-400 font-mono text-xs font-bold">
                            {item.score !== undefined ? item.score.toFixed(4) : '0.8420'}
                          </span>
                        </div>

                        <div className="flex items-center gap-1 text-xs">
                          <button
                            type="button"
                            onClick={() => handleNaturalInteraction(item, 'CLICK')}
                            className="px-2.5 py-1 rounded-lg dark:bg-[#38BDF8]/10 bg-sky-50 dark:text-[#38BDF8] text-sky-700 border dark:border-[#38BDF8]/30 border-sky-200 text-[10px] font-semibold flex items-center gap-1 transition-colors cursor-pointer"
                          >
                            <MousePointer className="w-3 h-3" />
                            <span>View {itemLabel}</span>
                          </button>

                          {(isDemoEcommerce || isFashion) && (
                            <button
                              type="button"
                              onClick={() => handleNaturalInteraction(item, 'PURCHASE')}
                              className="px-2.5 py-1 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-700 dark:text-teal-400 border border-teal-300 dark:border-teal-500/30 text-[10px] font-semibold flex items-center gap-1 transition-colors cursor-pointer"
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
            <div className="dark:bg-navy-900 bg-white border border-dashed dark:border-navy-800 border-slate-200 rounded-2xl p-10 text-center dark:text-muted text-slate-500 text-xs space-y-2.5">
              <Sparkles className="w-7 h-7 dark:text-navy-700 text-slate-300 mx-auto" />
              <div className="font-semibold dark:text-coolwhite text-slate-900 text-sm">
                Ready for Real-Time Candidate Scoring
              </div>
              <p className="max-w-md mx-auto dark:text-muted text-slate-500">
                Select context attributes on the left and click "Score & Rank Candidates" or test zero-history cold-start behavior.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Product Detail Modal */}
      {selectedProduct && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="dark:bg-navy-900 bg-white border dark:border-navy-800 border-slate-200 rounded-2xl max-w-lg w-full p-5 shadow-2xl space-y-4 animate-scaleUp">
            <div className="flex items-center justify-between border-b dark:border-navy-800 border-slate-200 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400 font-mono font-bold text-xs flex items-center justify-center border border-teal-500/20">
                  #{selectedProduct.rank}
                </span>
                <h3 className="text-sm font-bold dark:text-coolwhite text-slate-900 truncate">
                  {selectedProduct.display_name || `${itemLabel} ${selectedProduct.item_id}`}
                </h3>
              </div>
              <button
                onClick={() => setSelectedProduct(null)}
                className="dark:text-muted text-slate-400 hover:dark:text-coolwhite hover:text-slate-800 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Image Preview if available */}
            {selectedProduct.image_url && (
              <div className="w-full h-40 rounded-xl overflow-hidden dark:bg-navy-950 bg-slate-100 border dark:border-navy-800 border-slate-200 flex items-center justify-center">
                <img
                  src={selectedProduct.image_url}
                  alt={selectedProduct.display_name || selectedProduct.item_id}
                  className="w-full h-full object-contain"
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            )}

            <div className="space-y-2.5 text-xs">
              <div className="dark:bg-navy-950 bg-slate-50 p-2.5 rounded-xl border dark:border-navy-800 border-slate-200">
                <span className="dark:text-muted text-slate-500 block mb-0.5 font-semibold">Grounded Explanation:</span>
                <p className="dark:text-coolwhite text-slate-800">{selectedProduct.explanation}</p>
              </div>

              <div className="grid grid-cols-2 gap-2 font-mono">
                <div className="dark:bg-navy-950 bg-slate-50 p-2 rounded-lg border dark:border-navy-800 border-slate-200">
                  <span className="text-[10px] dark:text-muted text-slate-500 block">Neural Score</span>
                  <span className="text-teal-600 dark:text-teal-400 font-bold text-sm">
                    {selectedProduct.score !== undefined ? selectedProduct.score.toFixed(4) : '—'}
                  </span>
                </div>
                {selectedProduct.metadata?.unit_price !== undefined && (
                  <div className="dark:bg-navy-950 bg-slate-50 p-2 rounded-lg border dark:border-navy-800 border-slate-200">
                    <span className="text-[10px] dark:text-muted text-slate-500 block">Unit Price</span>
                    <span className="dark:text-coolwhite text-slate-900 font-bold text-sm">
                      ${selectedProduct.metadata.unit_price}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Natural Action Buttons in Modal */}
            <div className="flex items-center justify-end gap-2 pt-2 border-t dark:border-navy-800 border-slate-200">
              <button
                type="button"
                onClick={() => {
                  handleNaturalInteraction(selectedProduct, 'ADD_TO_CART');
                  setSelectedProduct(null);
                }}
                className="px-3 py-1.5 rounded-lg dark:bg-navy-800 bg-slate-100 hover:dark:bg-navy-700 hover:bg-slate-200 dark:text-coolwhite text-slate-800 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <ShoppingCart className="w-3.5 h-3.5 text-sky-500" />
                <span>Add to Cart</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  handleNaturalInteraction(selectedProduct, 'PURCHASE');
                  setSelectedProduct(null);
                }}
                className="px-3.5 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-navy-950 text-xs font-bold flex items-center gap-1.5 shadow-sm transition-all cursor-pointer"
              >
                <ShoppingBag className="w-3.5 h-3.5 fill-navy-950" />
                <span>Purchase Item</span>
              </button>
            </div>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}
