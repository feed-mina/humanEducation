'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import TrophyIcon from '@/components/assets/icons/ai/TrophyIcon';

interface AIChatGoalModalProps {
    show: boolean;
    onClose: () => void;
}

export default function AIChatGoalModal({ show, onClose }: AIChatGoalModalProps) {
    return (
        <AnimatePresence>
            {show && (
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="goal-celebration-overlay"
                >
                    <motion.div 
                        initial={{ scale: 0.7, y: 100, rotate: -5 }}
                        animate={{ scale: 1, y: 0, rotate: 0 }}
                        exit={{ scale: 0.5, opacity: 0 }}
                        className="goal-celebration-card"
                    >
                        <div className="trophy-emoji transform hover:scale-110 transition-transform">
                            <TrophyIcon />
                        </div>
                        <h2 className="tracking-tight">MISSION SUCCESS!</h2>
                        <div className="divider" />
                        
                        <p>
                            Amazing job! <br/>
                            You've reached <span className="highlight">10 Turns</span> today.<br/>
                            Consistency is the key to fluency.
                        </p>
                        
                        <motion.button 
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={onClose}
                            className="continue-btn"
                        >
                            Continue Learning
                        </motion.button>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
