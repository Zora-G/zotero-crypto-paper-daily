from .protocol import Paper
import math
import os
from urllib.parse import urlencode


def _cfg_get(cfg, key: str, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _build_feedback_url(endpoint: str, paper: Paper, action: str) -> str:
    params = {
        "action": action,
        "source": paper.source,
        "title": paper.title,
        "paper_url": paper.url,
    }
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}{urlencode(params)}"


def _build_feedback_buttons(paper: Paper, feedback_cfg) -> str:
    if not feedback_cfg:
        return ""
    if not _cfg_get(feedback_cfg, "enabled", False):
        return ""
    endpoint = os.getenv("FEEDBACK_ENDPOINT") or _cfg_get(feedback_cfg, "endpoint", None)
    if not endpoint:
        return ""

    like_url = _build_feedback_url(endpoint, paper, "liked")
    dislike_url = _build_feedback_url(endpoint, paper, "dislike")

    return (
        f' <a href="{like_url}" style="display: inline-block; text-decoration: none; '
        'font-size: 14px; font-weight: bold; color: #fff; background-color: #5cb85c; '
        'padding: 8px 16px; border-radius: 4px; margin-left: 8px;">推送满意</a>'
        f' <a href="{dislike_url}" style="display: inline-block; text-decoration: none; '
        'font-size: 14px; font-weight: bold; color: #fff; background-color: #aaa; '
        'padding: 8px 16px; border-radius: 4px; margin-left: 8px;">不太满意</a>'
    )


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(
    title: str,
    source: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: str = None,
    title_cn: str = "",
    source_note: str | None = None,
    feedback_buttons: str = "",
):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 4px 0 8px 0;">
            {title_cn}
        </td>
    </tr>
    <tr>
        <td style="font-size: 13px; color: #888; padding: 6px 0 2px 0;">
            <strong>Source:</strong> {source}{source_note_html}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 6px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>{feedback_buttons}
        </td>
    </tr>
</table>
"""
    translated_title = title_cn.strip() if isinstance(title_cn, str) and title_cn.strip() else title
    source_suffix = f" - {source_note}" if source_note else ""
    return block_template.format(
        title=title,
        source=source,
        source_note_html=source_suffix,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        title_cn=translated_title,
        feedback_buttons=feedback_buttons,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper], feedback_cfg=None) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        parts.append(
            get_block_html(
                p.title,
                p.source,
                authors,
                rate,
                p.tldr,
                p.pdf_url,
                affiliations,
                getattr(p, "title_cn", "") or "",
                p.source_note,
                feedback_buttons=_build_feedback_buttons(p, feedback_cfg),
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
