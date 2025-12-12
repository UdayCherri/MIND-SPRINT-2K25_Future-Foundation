import { motion } from 'motion/react';
import { useState } from 'react';
import { Page } from '../App';
import { Navigation } from '../components/Navigation';
import { User, Mail, Lock, Shield, Eye, Palette, Save, CheckCircle } from 'lucide-react';

interface SettingsPageProps {
  onNavigate: (page: Page) => void;
  onLogout: () => void;
}

export function SettingsPage({ onNavigate, onLogout }: SettingsPageProps) {
  const [fullName, setFullName] = useState('Alex Rivera');
  const [email, setEmail] = useState('alex.rivera@example.com');
  const [mbti, setMbti] = useState('INTJ');
  const [adapterLearning, setAdapterLearning] = useState(true);
  const [personalizedRecs, setPersonalizedRecs] = useState(true);
  const [localStorage, setLocalStorage] = useState(true);
  const [federatedUpdates, setFederatedUpdates] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [accentColor, setAccentColor] = useState('teal');
  const [saved, setSaved] = useState(false);

  const mbtiTypes = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
  ];

  const accentColors = [
    { name: 'teal', color: '#39E6C6', label: 'Teal' },
    { name: 'blue', color: '#3F77FF', label: 'Blue' },
    { name: 'purple', color: '#9B5CFF', label: 'Purple' },
    { name: 'pink', color: '#EC4899', label: 'Pink' },
  ];

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      <div className="relative z-10">
        <Navigation activeTab="Settings" onNavigate={onNavigate} onLogout={onLogout} />
        
        <main className="container mx-auto px-4 py-6 max-w-[1200px]">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl mb-2">Settings</h1>
            <p className="text-white/60">Customize your BAYMAX experience</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Profile Settings */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 rounded-xl bg-gradient-to-br from-teal-500 to-blue-500 shadow-lg shadow-teal-500/30">
                  <User className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-2xl">Profile Settings</h2>
              </div>

              <div className="space-y-4">
                {/* Full Name */}
                <div>
                  <label className="block text-sm text-white/70 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                  />
                </div>

                {/* Email */}
                <div>
                  <label className="block text-sm text-white/70 mb-2">Email Address</label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all"
                    />
                  </div>
                </div>

                {/* MBTI Type */}
                <div>
                  <label className="block text-sm text-white/70 mb-2">MBTI Type</label>
                  <select
                    value={mbti}
                    onChange={(e) => setMbti(e.target.value)}
                    className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white focus:outline-none focus:border-teal-400/50 focus:ring-2 focus:ring-teal-400/20 transition-all cursor-pointer"
                  >
                    {mbtiTypes.map((type) => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>

                {/* Change Password */}
                <button className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-white/10 text-white/70 hover:text-white hover:border-teal-400/50 transition-all flex items-center justify-center space-x-2">
                  <Lock className="w-4 h-4" />
                  <span>Change Password</span>
                </button>
              </div>
            </motion.div>

            {/* Personalization Settings */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 shadow-lg shadow-purple-500/30">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-2xl">Personalization</h2>
              </div>

              <div className="space-y-4">
                {/* Adapter Learning Toggle */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-white/5">
                  <div>
                    <div className="text-white/90 mb-1">Weekly Adapter Learning</div>
                    <div className="text-sm text-white/60">Continuously improve model accuracy</div>
                  </div>
                  <button
                    onClick={() => setAdapterLearning(!adapterLearning)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      adapterLearning ? 'bg-gradient-to-r from-teal-500 to-blue-500' : 'bg-slate-700'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
                      animate={{ x: adapterLearning ? 24 : 0 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                {/* Personalized Recommendations Toggle */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-white/5">
                  <div>
                    <div className="text-white/90 mb-1">Personalized Recommendations</div>
                    <div className="text-sm text-white/60">Get tailored wellness insights</div>
                  </div>
                  <button
                    onClick={() => setPersonalizedRecs(!personalizedRecs)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      personalizedRecs ? 'bg-gradient-to-r from-teal-500 to-blue-500' : 'bg-slate-700'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
                      animate={{ x: personalizedRecs ? 24 : 0 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Privacy Settings */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 shadow-lg shadow-blue-500/30">
                  <Eye className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-2xl">Privacy Settings</h2>
              </div>

              <div className="space-y-4">
                {/* Local Storage Toggle */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-white/5">
                  <div>
                    <div className="text-white/90 mb-1">Local Data Storage</div>
                    <div className="text-sm text-white/60">Store data locally on your device</div>
                  </div>
                  <button
                    onClick={() => setLocalStorage(!localStorage)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      localStorage ? 'bg-gradient-to-r from-teal-500 to-blue-500' : 'bg-slate-700'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
                      animate={{ x: localStorage ? 24 : 0 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                {/* Federated Updates Toggle */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-white/5">
                  <div>
                    <div className="text-white/90 mb-1">Federated Updates</div>
                    <div className="text-sm text-white/60">Contribute to model improvements</div>
                  </div>
                  <button
                    onClick={() => setFederatedUpdates(!federatedUpdates)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      federatedUpdates ? 'bg-gradient-to-r from-teal-500 to-blue-500' : 'bg-slate-700'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
                      animate={{ x: federatedUpdates ? 24 : 0 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-400/20">
                  <p className="text-sm text-white/70">
                    <span className="text-blue-400">Note:</span> Your data is encrypted and never shared without your explicit consent.
                  </p>
                </div>
              </div>
            </motion.div>

            {/* Display Settings */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="rounded-3xl bg-gradient-to-br from-slate-900/40 to-slate-800/40 backdrop-blur-xl border border-white/10 p-6 shadow-2xl"
            >
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/30">
                  <Palette className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-2xl">Display Settings</h2>
              </div>

              <div className="space-y-4">
                {/* Dark Mode Toggle */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-white/5">
                  <div>
                    <div className="text-white/90 mb-1">Dark Mode</div>
                    <div className="text-sm text-white/60">Use dark theme interface</div>
                  </div>
                  <button
                    onClick={() => setDarkMode(!darkMode)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      darkMode ? 'bg-gradient-to-r from-teal-500 to-blue-500' : 'bg-slate-700'
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-lg"
                      animate={{ x: darkMode ? 24 : 0 }}
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>

                {/* Accent Color Selector */}
                <div>
                  <label className="block text-sm text-white/70 mb-3">Accent Color Theme</label>
                  <div className="grid grid-cols-4 gap-3">
                    {accentColors.map((colorOption) => (
                      <button
                        key={colorOption.name}
                        onClick={() => setAccentColor(colorOption.name)}
                        className={`relative p-4 rounded-xl border-2 transition-all ${
                          accentColor === colorOption.name
                            ? 'border-white/30 bg-slate-800/50'
                            : 'border-white/10 bg-slate-800/30 hover:border-white/20'
                        }`}
                      >
                        <div
                          className="w-full h-12 rounded-lg shadow-lg"
                          style={{ backgroundColor: colorOption.color }}
                        />
                        <div className="text-sm text-white/70 mt-2 text-center">{colorOption.label}</div>
                        {accentColor === colorOption.name && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="absolute top-2 right-2"
                          >
                            <CheckCircle className="w-5 h-5 text-teal-400" />
                          </motion.div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Save Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-6"
          >
            <motion.button
              onClick={handleSave}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full px-6 py-4 rounded-xl bg-gradient-to-r from-teal-500 to-blue-500 text-white shadow-lg shadow-teal-500/30 hover:shadow-teal-500/50 transition-all flex items-center justify-center space-x-2 relative overflow-hidden group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              {saved ? (
                <>
                  <CheckCircle className="w-5 h-5 relative z-10" />
                  <span className="relative z-10">Changes Saved!</span>
                </>
              ) : (
                <>
                  <Save className="w-5 h-5 relative z-10" />
                  <span className="relative z-10">Save Changes</span>
                </>
              )}
            </motion.button>
          </motion.div>
        </main>
      </div>
    </div>
  );
}
