import React from 'react';

const TrafficLight = ({ isActive = false, color = 'red', size = 'medium' }) => {
  const sizeClasses = {
    small: 'w-3 h-3',
    medium: 'w-4 h-4',
    large: 'w-5 h-5'
  };

  const colorClasses = {
    red: 'bg-red-600',
    yellow: 'bg-yellow-500',
    green: 'bg-green-500'
  };

  return (
    <div 
      className={`${sizeClasses[size]} ${colorClasses[color]} rounded-full border-2 border-gray-600 transition-all duration-300 ${
        isActive 
          ? 'shadow-lg' 
          : 'opacity-20'
      }`}
      style={{
        boxShadow: isActive ? `0 0 15px ${color === 'red' ? '#dc2626' : color === 'yellow' ? '#eab308' : '#22c55e'}` : 'none'
      }}
    />
  );
};

export default TrafficLight;
