import math

def calculate_experience_score(years):
    """
    Computes a score based on a Gaussian distribution peaking around 7 years of experience.
    """
    ideal_years = 7.0
    std_dev = 2.0
    score = math.exp(-((years - ideal_years) ** 2) / (2 * (std_dev ** 2)))
    return score

def calculate_notice_period_score(notice_days):
    """
    Evaluates notice periods, giving a score of 1.0 for up to 30 days, decaying down to 0.0 for 180 days.
    """
    if notice_days <= 30:
        return 1.0
    elif notice_days >= 180:
        return 0.0
    else:
        return 1.0 - ((notice_days - 30) / 150.0)

def calculate_location_score(location):
    """
    Scores candidates based on their physical location, preferring Indian Tier-1 hubs specified in the job description.
    """
    location_lower = location.lower()
    preferred_hubs = ["pune", "noida", "delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"]
    if any(hub in location_lower for hub in preferred_hubs):
        return 1.0
    return 0.2

def calculate_platform_signals_score(signals):
    """
    Scores platform engagement signals such as response rates, activity, and completed assessments.
    """
    response_rate = signals.get('recruiter_response_rate', 0.0)
    interview_rate = signals.get('interview_completion_rate', 0.0)
    github_score = signals.get('github_activity_score', -1)
    if github_score == -1:
        normalized_github = 0.0
    else:
        normalized_github = github_score / 100.0
    avg_response_hours = signals.get('avg_response_time_hours', 200.0)
    response_speed_score = max(0.0, 1.0 - (avg_response_hours / 200.0))
    skill_scores = signals.get('skill_assessment_scores', {})
    assessment_sum = 0.0
    assessment_count = 0
    for key, val in skill_scores.items():
        assessment_sum += val
        assessment_count += 1
    assessment_score = (assessment_sum / (assessment_count * 100.0)) if assessment_count > 0 else 0.0
    final_score = (0.2 * response_rate + 0.2 * interview_rate + 0.2 * normalized_github + 0.2 * response_speed_score + 0.2 * assessment_score)
    return final_score

def compute_non_semantic_score(candidate):
    """
    Combines experience, notice, location, and platform signals into a single non-semantic weight score.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    years = profile.get('years_of_experience', 0.0)
    notice = signals.get('notice_period_days', 60)
    location = profile.get('location', '')
    exp_s = calculate_experience_score(years)
    notice_s = calculate_notice_period_score(notice)
    loc_s = calculate_location_score(location)
    sig_s = calculate_platform_signals_score(signals)
    overall_non_semantic = (0.35 * exp_s + 0.15 * notice_s + 0.15 * loc_s + 0.35 * sig_s)
    return overall_non_semantic
