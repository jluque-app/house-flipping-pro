import { useState, useEffect, useCallback } from 'react';
import { client } from '../api/client';
import type { FeatureCollection } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import axios from 'axios';

export function useProperties(bbox?: string, filters?: any) {
    const [data, setData] = useState<FeatureCollection | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { t } = useLanguage();

    const fetchProperties = useCallback(async (currentBbox: string) => {
        let isMounted = true;

        try {
            setLoading(true);
            const params = new URLSearchParams();
            if (currentBbox) params.append('bbox', currentBbox);

            // Add other filters
            if (filters) {
                Object.keys(filters).forEach(key => {
                    if (filters[key] !== undefined && filters[key] !== '') {
                        params.append(key, String(filters[key]));
                    }
                });
            }

            // Timeout implementation in 30 seconds (increased from 15s)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);

            const res = await client.get<FeatureCollection>(`/properties?${params.toString()}`, {
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (isMounted) {
                setData(res.data);
                setError(null);
            }
        } catch (err: any) {
            console.error(err);
            if (isMounted) {
                // Check for cancellation (timeout)
                if (axios.isCancel(err) || err.code === 'ECONNABORTED' || err.name === 'CanceledError') {
                    setError(t('error.timeout'));
                } else {
                    setError(t('error.general'));
                }
            }
        } finally {
            if (isMounted) setLoading(false);
        }
    }, [filters, t]);

    // Debounced fetch
    useEffect(() => {
        if (!bbox) return;

        const timeoutId = setTimeout(() => {
            fetchProperties(bbox);
        }, 500); // 500ms debounce

        return () => clearTimeout(timeoutId);
    }, [bbox, fetchProperties]);

    return { data, loading, error };
}
