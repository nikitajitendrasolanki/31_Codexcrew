import React, { useState, useEffect } from 'react';

const NumberTicker = ({ 
  value, 
  duration = 2000, 
  className = '', 
  prefix = '', 
  suffix = '',
  startValue = 0 
}) => {
  const [displayValue, setDisplayValue] = useState(startValue);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    setIsAnimating(true);
    
    // Handle percentage values
    const numericValue = typeof value === 'string' && value.includes('%') 
      ? parseFloat(value.replace('%', '')) 
      : typeof value === 'number' ? value : parseFloat(value);
    
    if (isNaN(numericValue)) {
      setDisplayValue(value);
      setIsAnimating(false);
      return;
    }

    const startTime = Date.now();
    const difference = numericValue - startValue;
    
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Smooth easing function
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (difference * easeOutCubic);
      
      // For decimal values, show decimal places during animation
      if (numericValue % 1 !== 0) {
        setDisplayValue(parseFloat(currentValue.toFixed(1)));
      } else {
        setDisplayValue(Math.floor(currentValue));
      }
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setDisplayValue(numericValue);
        setIsAnimating(false);
      }
    };
    
    // Small delay to ensure smooth start
    const timer = setTimeout(() => {
      requestAnimationFrame(animate);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [value, duration, startValue]);

  const formatValue = (val) => {
    // Check if original value was a percentage
    const isPercentage = typeof value === 'string' && value.includes('%');
    
    if (isPercentage) {
      return `${val}%`;
    }
    return typeof val === 'number' ? val.toLocaleString() : val;
  };

  return (
    <span className={`transition-all duration-300 ${isAnimating ? 'scale-105' : 'scale-100'} ${className}`}>
      {prefix}{formatValue(displayValue)}{suffix}
    </span>
  );
};

export default NumberTicker;
