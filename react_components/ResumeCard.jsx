/**
 * ResumeCard Component - Tailwind CSS
 * 
 * A reusable React component for displaying resume evaluation results
 * with score badges, progress indicators, and detailed breakdowns.
 * 
 * Props:
 * - resume: Resume object with evaluation data
 * - onViewDetails: Callback when "View Details" is clicked
 * - onExport: Callback when "Export" is clicked
 * - showActions: Boolean to show/hide action buttons
 */

import React from 'react';
import { 
  UserIcon, 
  DocumentIcon, 
  CalendarIcon,
  ChevronRightIcon,
  DownloadIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon 
} from '@heroicons/react/24/outline';

const ResumeCard = ({ 
  resume, 
  onViewDetails, 
  onExport, 
  showActions = true,
  className = ""
}) => {
  // Helper function to get verdict styling
  const getVerdictStyling = (verdict, score) => {
    switch (verdict) {
      case 'high':
        return {
          badge: 'bg-green-100 text-green-800 border-green-200',
          progress: 'bg-green-500',
          icon: <CheckCircleIcon className="w-4 h-4 text-green-600" />,
          label: `High (${score}/100)`
        };
      case 'medium':
        return {
          badge: 'bg-yellow-100 text-yellow-800 border-yellow-200',
          progress: 'bg-yellow-500', 
          icon: <ExclamationTriangleIcon className="w-4 h-4 text-yellow-600" />,
          label: `Medium (${score}/100)`
        };
      case 'low':
        return {
          badge: 'bg-red-100 text-red-800 border-red-200',
          progress: 'bg-red-500',
          icon: <ExclamationTriangleIcon className="w-4 h-4 text-red-600" />,
          label: `Low (${score}/100)`
        };
      default:
        return {
          badge: 'bg-gray-100 text-gray-800 border-gray-200',
          progress: 'bg-gray-400',
          icon: <DocumentIcon className="w-4 h-4 text-gray-600" />,
          label: 'Not Evaluated'
        };
    }
  };

  const {
    resume_id,
    candidate_name,
    filename,
    final_score,
    verdict, 
    jd_title,
    jd_company,
    created_at,
    missing_elements = [],
    feedback_count = 0
  } = resume;

  const verdictStyle = getVerdictStyling(verdict, final_score);
  const scorePercentage = final_score ? (final_score / 100) * 100 : 0;
  const formattedDate = created_at ? new Date(created_at).toLocaleDateString() : 'Unknown';

  return (
    <div 
      className={`bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200 ${className}`}
      role="article"
      aria-labelledby={`resume-${resume_id}-title`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-start justify-between">
          {/* Candidate Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2">
              <UserIcon className="w-5 h-5 text-gray-400 flex-shrink-0" />
              <h3 
                id={`resume-${resume_id}-title`}
                className="text-lg font-semibold text-gray-900 truncate"
              >
                {candidate_name || 'Unknown Candidate'}
              </h3>
            </div>
            
            <div className="mt-1 flex items-center space-x-4 text-sm text-gray-500">
              <div className="flex items-center space-x-1">
                <DocumentIcon className="w-4 h-4" />
                <span className="truncate max-w-48" title={filename}>
                  {filename}
                </span>
              </div>
              
              <div className="flex items-center space-x-1">
                <CalendarIcon className="w-4 h-4" />
                <span>{formattedDate}</span>
              </div>
            </div>
          </div>

          {/* Score Badge */}
          <div className="flex-shrink-0 ml-4">
            <div 
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${verdictStyle.badge}`}
              role="status"
              aria-label={`Evaluation score: ${verdictStyle.label}`}
            >
              {verdictStyle.icon}
              <span className="ml-1">{verdictStyle.label}</span>
            </div>
          </div>
        </div>

        {/* Job Application Info */}
        {jd_title && (
          <div className="mt-3 p-2 bg-gray-50 rounded-md">
            <p className="text-sm text-gray-700">
              <span className="font-medium">Applied for:</span>{' '}
              {jd_title} {jd_company && `at ${jd_company}`}
            </p>
          </div>
        )}
      </div>

      {/* Score Progress & Details */}
      {final_score !== null && (
        <div className="p-4 space-y-3">
          {/* Progress Bar */}
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Match Score</span>
              <span className="font-medium">{final_score}/100</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${verdictStyle.progress}`}
                style={{ width: `${scorePercentage}%` }}
                role="progressbar"
                aria-valuenow={final_score}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-label={`Match score: ${final_score} out of 100`}
              />
            </div>
          </div>

          {/* Quick Insights */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            {/* Missing Elements */}
            <div>
              <span className="text-gray-600">Missing Skills:</span>
              <span className="ml-2 font-medium text-red-600">
                {missing_elements.length > 0 ? missing_elements.length : 'None'}
              </span>
            </div>

            {/* Feedback Available */}
            <div>
              <span className="text-gray-600">Suggestions:</span>
              <span className="ml-2 font-medium text-blue-600">
                {feedback_count > 0 ? `${feedback_count} available` : 'None'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {showActions && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-100 flex justify-between items-center">
          <button
            onClick={() => onViewDetails?.(resume)}
            className="inline-flex items-center px-3 py-2 text-sm font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 transition-colors duration-200"
            aria-label={`View detailed evaluation for ${candidate_name || 'this candidate'}`}
          >
            View Details
            <ChevronRightIcon className="ml-1 w-4 h-4" />
          </button>

          {final_score && final_score >= 70 && (
            <button
              onClick={() => onExport?.(resume)}
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 transition-colors duration-200"
              aria-label={`Export resume for ${candidate_name || 'this candidate'}`}
            >
              <DownloadIcon className="mr-1 w-4 h-4" />
              Export
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// PropTypes for better development experience
ResumeCard.propTypes = {
  resume: PropTypes.shape({
    resume_id: PropTypes.string.isRequired,
    candidate_name: PropTypes.string,
    filename: PropTypes.string.isRequired,
    final_score: PropTypes.number,
    verdict: PropTypes.oneOf(['high', 'medium', 'low']),
    jd_title: PropTypes.string,
    jd_company: PropTypes.string,
    created_at: PropTypes.string,
    missing_elements: PropTypes.array,
    feedback_count: PropTypes.number
  }).isRequired,
  onViewDetails: PropTypes.func,
  onExport: PropTypes.func,
  showActions: PropTypes.bool,
  className: PropTypes.string
};

// Example usage and mock data for development
export const ResumeCardExample = () => {
  const mockResume = {
    resume_id: "resume-123",
    candidate_name: "Sarah Johnson",
    filename: "sarah_johnson_resume.pdf", 
    final_score: 87,
    verdict: "high",
    jd_title: "Senior Python Developer",
    jd_company: "TechCorp Inc",
    created_at: "2024-01-15T10:30:00Z",
    missing_elements: ["docker", "kubernetes"],
    feedback_count: 3
  };

  const handleViewDetails = (resume) => {
    console.log('View details for:', resume.resume_id);
    // Navigate to detailed view
  };

  const handleExport = (resume) => {
    console.log('Export resume:', resume.resume_id);
    // Trigger CSV/PDF export
  };

  return (
    <div className="max-w-md mx-auto p-4">
      <h2 className="text-xl font-bold mb-4">Resume Card Example</h2>
      <ResumeCard
        resume={mockResume}
        onViewDetails={handleViewDetails}
        onExport={handleExport}
      />
    </div>
  );
};

export default ResumeCard;

/*
Usage in a list/grid:

import ResumeCard from './components/ResumeCard';

const ResumeList = ({ resumes }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {resumes.map(resume => (
        <ResumeCard
          key={resume.resume_id}
          resume={resume}
          onViewDetails={handleViewDetails}
          onExport={handleExport}
        />
      ))}
    </div>
  );
};

Required Tailwind CSS classes:
- Ensure Heroicons is installed: npm install @heroicons/react
- Tailwind CSS with default configuration
- Focus and hover states included for accessibility
*/
