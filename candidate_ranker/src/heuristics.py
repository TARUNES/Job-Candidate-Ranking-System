import datetime

def has_timeline_overlaps(career_history):
    """
    Scans a candidate's career history list to verify if there are impossible overlapping full-time employment intervals exceeding 90 days.
    """
    intervals = []
    for job in career_history:
        start_str = job.get('start_date')
        end_str = job.get('end_date')
        if not start_str:
            continue
        try:
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
            if end_str:
                end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
            else:
                end_dt = datetime.date.today()
            intervals.append((start_dt, end_dt))
        except ValueError:
            continue
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        current_end = intervals[i][1]
        next_start = intervals[i+1][0]
        if current_end > next_start:
            overlap_days = (current_end - next_start).days
            if overlap_days > 90:
                return True
    return False

def has_impossible_skills(skills):
    """
    Checks for impossible skill claims where a candidate claims 'expert' or 'advanced' proficiency but has 0 duration months listed.
    """
    expert_advanced_count = 0
    zero_duration_count = 0
    for s in skills:
        proficiency = s.get('proficiency', '').lower()
        duration = s.get('duration_months', 0)
        if proficiency in ['expert', 'advanced']:
            expert_advanced_count += 1
            if duration == 0:
                zero_duration_count += 1
    if expert_advanced_count >= 5 and zero_duration_count >= 3:
        return True
    return False

def is_pure_consulting_career(career_history):
    """
    Checks if a candidate's entire career has been spent solely at consulting or outsourcing firms, which is a strict disqualifier in the job description.
    """
    consulting_keywords = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy", "wipro technologies", "infosys technologies"]
    if not career_history:
        return False
    all_consulting = True
    for job in career_history:
        company_name = job.get('company', '').lower()
        is_consulting = any(keyword in company_name for keyword in consulting_keywords)
        if not is_consulting:
            all_consulting = False
            break
    return all_consulting

def is_mismatched_profile(headline, summary, career_history):
    """
    Flags candidates who are behavioral twins or have completely mismatched titles and responsibilities (e.g., HR Manager with accounting descriptions).
    """
    headline_lower = headline.lower()
    summary_lower = summary.lower()
    if "hr manager" in headline_lower or "hr manager" in summary_lower:
        for job in career_history:
            desc = job.get('description', '').lower()
            if "accounting" in desc or "financial reporting" in desc or "gaap" in desc:
                return True
    if "operations manager" in headline_lower or "operations manager" in summary_lower:
        for job in career_history:
            desc = job.get('description', '').lower()
            if "mechanical engineering" in desc or "solidworks" in desc:
                return True
    if "customer support" in headline_lower or "customer support" in summary_lower:
        for job in career_history:
            desc = job.get('description', '').lower()
            if "business analyst" in desc or "retail and cpg" in desc:
                return True
    return False

def check_honeypots_and_filters(candidate):
    """
    Aggregates all check sub-routines to determine if a candidate is a trap/honeypot or violates hard disqualification rules.
    """
    profile = candidate.get('profile', {})
    career_history = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    if is_pure_consulting_career(career_history):
        return True
    if has_timeline_overlaps(career_history):
        return True
    if has_impossible_skills(skills):
        return True
    if is_mismatched_profile(headline, summary, career_history):
        return True
    return False
