#!/usr/bin/env python
# core/push_wechat.py
"""
微信推送通知：模拟盘交易、每日信号、异常告警
免费方案：PushPlus (www.pushplus.plus) — 注册获取 token 即可
"""
import requests

def push_wechat(title, content, token=None):
    """
    发送微信通知
    token: PushPlus token（注册 pushplus.plus 获取）
    """
    if token is None:
        from config.settings import PUSHPLUS_TOKEN
        token = PUSHPLUS_TOKEN

    if not token or token.startswith("你的"):
        return False

    try:
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json={
                "token": token,
                "title": title,
                "content": content,
                "template": "txt",
            },
            timeout=5,
        )
        return resp.json().get("code") == 200
    except Exception:
        return False


def notify_trade(action, code, price, shares, profit, detail=""):
    """交易通知"""
    emoji = "🟢" if action == "买入" else ("✅" if profit > 0 else "❌")
    content = f"{emoji} {action} {code}\n"
    content += f"价格: {price:.2f} 数量: {shares}股\n"
    content += f"盈亏: {profit*100:+.1f}%\n" if action != "买入" else ""
    content += f"理由: {detail}" if detail else ""
    push_wechat(f"{action} {code}", content)


def notify_daily_summary(buy_count, hold_count, watch_count, regime=""):
    """每日信号汇总"""
    content = f"买入 {buy_count} | 关注 {watch_count} | 观望 {hold_count}"
    if regime:
        content += f"\n市场状态: {regime}"
    push_wechat("每日选股信号", content)


def notify_alert(message):
    """异常告警"""
    push_wechat("⚠ 系统告警", message)
