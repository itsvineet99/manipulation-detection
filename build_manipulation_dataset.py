from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

OUTPUT_PATH = Path("data/manipulation_detection_dataset.csv")
DEFAULT_ROWS = 10_000
DEFAULT_SEED = 42

INTENT_METADATA = {
    "request_review": {
        "intent": "Request review before proceeding",
        "timeframes": ["before tomorrow morning", "before the next meeting"],
        "reason": "I can correct anything important before we move ahead.",
    },
    "confirm_attendance": {
        "intent": "Confirm availability or attendance",
        "timeframes": ["by tonight", "before the schedule is finalized"],
        "reason": "I can lock the schedule without following up repeatedly.",
    },
    "share_update": {
        "intent": "Share a current update",
        "timeframes": ["before end of day", "after it changes"],
        "reason": "everyone should have the latest status in one place.",
    },
    "follow_process": {
        "intent": "Follow the required process",
        "timeframes": ["this time", "for the next submission"],
        "reason": "it keeps the work aligned with the expected process.",
    },
    "share_materials": {
        "intent": "Send the needed materials",
        "timeframes": ["today", "as soon as it is ready"],
        "reason": "the next step depends on having the material available.",
    },
    "raise_concern": {
        "intent": "Raise a concern early",
        "timeframes": ["before we commit", "while there is still time to adjust"],
        "reason": "it is easier to fix concerns early than later.",
    },
    "ask_for_help": {
        "intent": "Ask for practical help",
        "timeframes": ["when you have a moment", "so we can finish on time"],
        "reason": "the workload is easier to manage when it is shared clearly.",
    },
    "request_privacy": {
        "intent": "Keep information private",
        "timeframes": ["for now", "until we decide what to share"],
        "reason": "the information is sensitive and not ready for broad sharing.",
    },
    "discuss_budget": {
        "intent": "Discuss cost or budget constraints",
        "timeframes": ["before we lock anything in", "while we can still adjust the plan"],
        "reason": "we should catch budget issues before money is committed.",
    },
    "delay_until_ready": {
        "intent": "Wait until prerequisites are ready",
        "timeframes": ["for now", "until the missing piece is confirmed"],
        "reason": "moving too early would create unnecessary rework.",
    },
    "safety_check": {
        "intent": "Check on safety or wellbeing",
        "timeframes": ["right away", "as soon as you notice it"],
        "reason": "we need to respond quickly if something changes.",
    },
    "give_feedback": {
        "intent": "Give honest feedback or clarification",
        "timeframes": ["when you read it", "before I send the final version"],
        "reason": "clear feedback now prevents confusion later.",
    },
}

TACTICS = [
    "guilt",
    "urgency",
    "shame",
    "fear",
    "reciprocity",
    "authority_pressure",
    "social_proof",
    "blame_shifting",
    "emotional_blackmail",
    "false_dichotomy",
    "financial_pressure",
    "threat",
    "flattery_pressure",
    "minimization",
    "isolation",
]

TACTIC_SEVERITY = {
    "guilt": "low",
    "urgency": "low",
    "shame": "medium",
    "fear": "medium",
    "reciprocity": "medium",
    "authority_pressure": "medium",
    "social_proof": "low",
    "blame_shifting": "medium",
    "emotional_blackmail": "high",
    "false_dichotomy": "medium",
    "financial_pressure": "high",
    "threat": "high",
    "flattery_pressure": "low",
    "minimization": "low",
    "isolation": "high",
}

NEUTRAL_TEMPLATES = [
    "Could you {action} {timeframe} so {reason}",
    "Please {action} {timeframe}; {reason}",
    "I would like you to {action} {timeframe} because {reason}",
    "If possible, {action} {timeframe}; {reason}",
    "Let me know if you can {action} {timeframe}, because {reason}",
]

MANIPULATIVE_TEMPLATES = {
    "guilt": [
        "If you really cared, you would {action} {timeframe} instead of leaving the rest of this on other people.",
        "You should {action} {timeframe}; otherwise it feels like you are comfortable making this harder for everyone else.",
    ],
    "urgency": [
        "You need to {action} {timeframe} right now, because waiting any longer will only make this worse.",
        "Do not delay this any further; {action} {timeframe} before the chance to fix it disappears.",
    ],
    "shame": [
        "Only someone careless would refuse to {action} {timeframe} after how clearly this has been explained.",
        "At this point, not choosing to {action} {timeframe} would just look irresponsible.",
    ],
    "fear": [
        "If you do not {action} {timeframe}, this will turn into a bigger problem and people will remember who slowed it down.",
        "You need to {action} {timeframe}, or this is going to create a mess that comes back on you.",
    ],
    "reciprocity": [
        "After all the time I have spent helping you, the least you can do is {action} {timeframe} without pushing back.",
        "I have shown up for you before, so you should {action} {timeframe} instead of making me ask twice.",
    ],
    "authority_pressure": [
        "Just {action} {timeframe} and stop questioning it, because this is already the direction I decided on.",
        "You do not need to debate this further; {action} {timeframe} because I am telling you it needs to happen.",
    ],
    "social_proof": [
        "Everyone else is already aligned on this, so you need to {action} {timeframe} instead of being the one person holding it up.",
        "People who understand the situation have already agreed to {action}, so do not make yourself the exception {timeframe}.",
    ],
    "blame_shifting": [
        "If this falls apart, it will be because you did not {action} {timeframe} when there was still time to fix it.",
        "Remember that any problem from here will trace back to your choice not to {action} {timeframe}.",
    ],
    "emotional_blackmail": [
        "If you actually cared about me, you would {action} {timeframe} instead of making me ask again.",
        "Someone who truly supports me would {action} {timeframe} without turning it into a negotiation.",
    ],
    "false_dichotomy": [
        "Either you {action} {timeframe}, or you are making it clear that you are not on our side.",
        "There really are only two options here: {action} {timeframe}, or admit that you are choosing not to support this.",
    ],
    "financial_pressure": [
        "If you do not {action} {timeframe}, expect the extra cost and delay to come back to you.",
        "You need to {action} {timeframe}, unless you are comfortable being blamed for the money this wastes.",
    ],
    "threat": [
        "If you do not {action} {timeframe}, I will make sure the right people know exactly who caused the problem.",
        "Choose whether you want to {action} {timeframe}, because if not I will escalate this directly.",
    ],
    "flattery_pressure": [
        "You are the only person capable of handling this properly, so you need to {action} {timeframe} for everyone's sake.",
        "I am asking you because nobody else is reliable enough, which means you should {action} {timeframe}.",
    ],
    "minimization": [
        "Stop overthinking it and just {action} {timeframe}; this is simple and it should not need more discussion.",
        "You are making this bigger than it is, so just {action} {timeframe} and move on.",
    ],
    "isolation": [
        "Do not run this by anyone else; just {action} {timeframe}, because other people will only interfere and confuse things.",
        "Keep this between us and {action} {timeframe}; bringing in others will only complicate what should be straightforward.",
    ],
}

DOMAIN_ACTIONS = {
    "workplace": {
        "request_review": [
            "review the client deck in the shared drive",
            "look over the vendor summary before procurement meets",
        ],
        "confirm_attendance": [
            "confirm whether you can join the launch call",
            "tell me if you can attend the handoff meeting",
        ],
        "share_update": [
            "post the latest status in the sprint tracker",
            "update the risk log after the standup",
        ],
        "follow_process": [
            "use the current expense template for the reimbursement",
            "submit the request through the approval workflow",
        ],
        "share_materials": [
            "send the signed purchase order once procurement approves it",
            "share the meeting notes from the client workshop",
        ],
        "raise_concern": [
            "tell me if the rollout timeline feels risky",
            "flag anything unclear in the vendor contract",
        ],
        "ask_for_help": [
            "help me merge the presentation slides",
            "help cover the budget cleanup before finance reviews it",
        ],
        "request_privacy": [
            "keep the pricing draft within the leadership group",
            "not share the headcount plan outside the hiring team",
        ],
        "discuss_budget": [
            "tell me if the travel budget looks too high",
            "let me know if the vendor quote exceeds what we expected",
        ],
        "delay_until_ready": [
            "wait until legal clears the wording before publishing the announcement",
            "hold the rollout until QA signs off on the release notes",
        ],
        "safety_check": [
            "tell facilities if the monitor cable keeps overheating",
            "report any issue if the loading dock door sticks again",
        ],
        "give_feedback": [
            "tell me if the meeting summary still feels incomplete",
            "let me know if the handoff instructions are unclear",
        ],
    },
    "family": {
        "request_review": [
            "review the school form before I submit it",
            "look over the trip itinerary before I book anything",
        ],
        "confirm_attendance": [
            "tell me if you can make the family dinner",
            "confirm whether you can attend the weekend reunion",
        ],
        "share_update": [
            "post the latest change in the family calendar",
            "send an update in the family group chat when plans shift",
        ],
        "follow_process": [
            "follow the medicine schedule on the fridge",
            "use the grocery list on the kitchen board",
        ],
        "share_materials": [
            "send me the signed permission slip",
            "bring the insurance receipt after the appointment",
        ],
        "raise_concern": [
            "tell me if the holiday plan feels too expensive",
            "let me know if the bedtime schedule seems unrealistic",
        ],
        "ask_for_help": [
            "help with grocery pickup on your way home",
            "take the kids for an hour while I finish the doctor call",
        ],
        "request_privacy": [
            "keep the health update within the family",
            "not share the money discussion with the relatives",
        ],
        "discuss_budget": [
            "tell me if the repair bill feels too high",
            "let me know if the holiday budget needs to change",
        ],
        "delay_until_ready": [
            "wait until the school confirms the schedule before booking tickets",
            "hold off on inviting everyone until the doctor calls back",
        ],
        "safety_check": [
            "text me when you get home from the party",
            "tell me right away if the fever gets worse tonight",
        ],
        "give_feedback": [
            "tell me if the weekend plan feels too packed",
            "let me know if that joke in the group chat crossed a line",
        ],
    },
    "friendship": {
        "request_review": [
            "review the trip plan before I reserve the rental car",
            "look over the dinner options before I book a table",
        ],
        "confirm_attendance": [
            "tell me if you are joining the camping trip",
            "confirm whether you can make the birthday dinner",
        ],
        "share_update": [
            "post the latest plan in the group chat",
            "send the final timing in the shared note once you know it",
        ],
        "follow_process": [
            "use the shared expense sheet when you log costs",
            "put the photos in the album link instead of the thread",
        ],
        "share_materials": [
            "send your part of the birthday video tonight",
            "share the playlist link before the road trip",
        ],
        "raise_concern": [
            "tell me if inviting Alex will make the group awkward",
            "let me know if the bill split looks unfair",
        ],
        "ask_for_help": [
            "help carry the supplies to the picnic spot",
            "help me pick up the board games before people arrive",
        ],
        "request_privacy": [
            "keep what I told you about the breakup private",
            "not post the surprise dinner plans yet",
        ],
        "discuss_budget": [
            "tell me if the cabin price is too much",
            "let me know if the ticket cost no longer works for you",
        ],
        "delay_until_ready": [
            "wait until everyone votes before booking the restaurant",
            "hold off on announcing the plan until Maya confirms",
        ],
        "safety_check": [
            "text me when you get home after the concert",
            "let me know if your ride falls through tonight",
        ],
        "give_feedback": [
            "tell me if the group chat tone feels off",
            "let me know if the weekend itinerary is confusing",
        ],
    },
    "school": {
        "request_review": [
            "review my lab draft before I submit it",
            "look over the group presentation slides before class",
        ],
        "confirm_attendance": [
            "tell me if you can make the study session",
            "confirm whether you can attend the tutoring slot",
        ],
        "share_update": [
            "post your progress in the project doc",
            "update the research notes after you finish the article",
        ],
        "follow_process": [
            "use the rubric when you format the paper",
            "submit the form through the course portal",
        ],
        "share_materials": [
            "send your source summaries for the literature review",
            "bring the signed field-trip permission slip tomorrow",
        ],
        "raise_concern": [
            "tell me if the workload feels unrealistic",
            "let me know if the team instructions seem unfair",
        ],
        "ask_for_help": [
            "help me check the citation formatting",
            "help cover my section in the study group if I run late",
        ],
        "request_privacy": [
            "keep the accommodation request private",
            "not share the exam accommodation letter",
        ],
        "discuss_budget": [
            "tell me if the textbook cost is a problem",
            "let me know if the lab fee is too high this term",
        ],
        "delay_until_ready": [
            "wait until the professor confirms the rubric change",
            "hold off on submitting until you verify the course code",
        ],
        "safety_check": [
            "tell the instructor if the lab equipment sparks again",
            "let me know if you feel sick during the field trip",
        ],
        "give_feedback": [
            "tell me if the thesis statement feels weak",
            "let me know if the lecture summary is unclear",
        ],
    },
    "sales": {
        "request_review": [
            "review the proposal before we send it to procurement",
            "look over the pricing page before the client demo",
        ],
        "confirm_attendance": [
            "tell me if you can join the discovery call",
            "confirm whether the buyer can attend the final demo",
        ],
        "share_update": [
            "send the latest account update after the call",
            "post the pipeline notes in the CRM before lunch",
        ],
        "follow_process": [
            "use the approved pricing sheet during the quote review",
            "route the redlines through legal before promising changes",
        ],
        "share_materials": [
            "send the revised quote after finance signs off",
            "share the pilot success criteria with the buyer committee",
        ],
        "raise_concern": [
            "tell me if the onboarding timeline feels risky",
            "let me know if any contract term looks unrealistic",
        ],
        "ask_for_help": [
            "help prep the demo environment before the meeting",
            "help me compare the support tiers for the client",
        ],
        "request_privacy": [
            "keep the discount discussion inside the buying team",
            "not share the draft contract outside legal review",
        ],
        "discuss_budget": [
            "tell me if the rollout cost is beyond the budget",
            "let me know if the annual price still blocks the deal",
        ],
        "delay_until_ready": [
            "wait until both legal teams finish redlines before announcing anything",
            "hold the partnership post until the contract is signed",
        ],
        "safety_check": [
            "tell me if the trial environment exposes live customer data",
            "report it immediately if the demo account breaks permissions",
        ],
        "give_feedback": [
            "tell me if the proposal language feels too vague",
            "let me know if the recap email misses a key objection",
        ],
    },
    "healthcare": {
        "request_review": [
            "review the care plan before the follow-up visit",
            "look over the discharge instructions before you leave the clinic",
        ],
        "confirm_attendance": [
            "tell me if you can make the therapy appointment",
            "confirm whether you can attend the follow-up scan",
        ],
        "share_update": [
            "send an update if the symptoms change overnight",
            "post the medication changes in the patient portal",
        ],
        "follow_process": [
            "follow the dosing instructions on the prescription label",
            "use the check-in form before the telehealth call",
        ],
        "share_materials": [
            "bring the insurance letter to the appointment",
            "send the latest medication list before the consult",
        ],
        "raise_concern": [
            "tell the clinician if the pain gets sharper",
            "let me know if the recovery plan feels too aggressive",
        ],
        "ask_for_help": [
            "help me track the medication times tonight",
            "come with me to the appointment if you are available",
        ],
        "request_privacy": [
            "keep the diagnosis within close family",
            "not post the treatment update online yet",
        ],
        "discuss_budget": [
            "tell me if the procedure estimate feels too high",
            "let me know if the copay is a barrier this month",
        ],
        "delay_until_ready": [
            "wait for the test results before changing the treatment plan",
            "hold off on new supplements until the clinician calls back",
        ],
        "safety_check": [
            "call the office if the fever rises again tonight",
            "tell the nurse right away if dizziness gets worse",
        ],
        "give_feedback": [
            "tell the therapist if the exercise pace feels overwhelming",
            "let me know if the care instructions are unclear",
        ],
    },
    "community": {
        "request_review": [
            "review the event flyer before we print it",
            "look over the volunteer schedule before I send it out",
        ],
        "confirm_attendance": [
            "tell me if you can volunteer at the cleanup",
            "confirm whether your group can join the town hall",
        ],
        "share_update": [
            "post the latest numbers in the fundraiser tracker",
            "send the setup update in the organizer chat",
        ],
        "follow_process": [
            "use the approved talking points during outreach",
            "follow the safety checklist before the event opens",
        ],
        "share_materials": [
            "send the donor pledge sheet to the treasurer",
            "bring the sign-in sheets for the meeting",
        ],
        "raise_concern": [
            "tell me if the fundraising goal feels unrealistic",
            "let me know if the site layout seems unsafe",
        ],
        "ask_for_help": [
            "help set up the water station before people arrive",
            "help me count the supplies after the event",
        ],
        "request_privacy": [
            "keep the donor list private until the report is posted",
            "not share the complaint details outside the review team",
        ],
        "discuss_budget": [
            "tell me if the equipment rental cost is too high",
            "let me know if the printing budget needs to change",
        ],
        "delay_until_ready": [
            "wait until the permits clear before posting the schedule",
            "hold off on sharing photos until parents approve them",
        ],
        "safety_check": [
            "tell me if anyone needs accessibility support at the venue",
            "report it right away if the extension cables create a tripping hazard",
        ],
        "give_feedback": [
            "tell me if the meeting notes missed anything important",
            "let me know if the outreach message sounds too aggressive",
        ],
    },
    "housing": {
        "request_review": [
            "review the lease addendum before signing it",
            "look over the move-out checklist before the inspection",
        ],
        "confirm_attendance": [
            "tell me if you can meet the maintenance crew tomorrow",
            "confirm whether you can join the walkthrough on Friday",
        ],
        "share_update": [
            "send an update if the repair window changes",
            "post the latest building notice in the resident chat",
        ],
        "follow_process": [
            "use the resident portal for maintenance requests",
            "follow the trash pickup rules in the lease",
        ],
        "share_materials": [
            "send photos of the water damage today",
            "share the updated mailing address after the move",
        ],
        "raise_concern": [
            "tell me if the roommate agreement feels unclear",
            "let me know if the parking policy seems unfair",
        ],
        "ask_for_help": [
            "help move the boxes before the inspection",
            "help me coordinate the key handoff with the new tenant",
        ],
        "request_privacy": [
            "keep the entry code private",
            "not share the landlord draft notice yet",
        ],
        "discuss_budget": [
            "tell me if the rent increase is too steep",
            "let me know if the repair quote seems unreasonable",
        ],
        "delay_until_ready": [
            "wait until the inspection is done before moving furniture into the hall",
            "hold off on signing until the landlord answers the repair questions",
        ],
        "safety_check": [
            "tell me if the stairwell light fails again",
            "report it immediately if the front door lock sticks tonight",
        ],
        "give_feedback": [
            "tell me if the maintenance update feels vague",
            "let me know if the roommate note sounds too harsh",
        ],
    },
    "online": {
        "request_review": [
            "review the post draft before it goes live",
            "look over the moderation rules before I pin them",
        ],
        "confirm_attendance": [
            "tell me if you can join the livestream test",
            "confirm whether you can help moderate the event chat",
        ],
        "share_update": [
            "post the latest status in the support thread",
            "send the final asset update in the content channel",
        ],
        "follow_process": [
            "use the support form for billing issues",
            "follow the server rules before inviting new members",
        ],
        "share_materials": [
            "send the updated image files to the drive",
            "share the login screenshot through the secure form",
        ],
        "raise_concern": [
            "tell me if the thread pace feels overwhelming",
            "let me know if the new rule sounds unclear",
        ],
        "ask_for_help": [
            "help clear the comment queue tonight",
            "help me tag the post correctly before publication",
        ],
        "request_privacy": [
            "keep my new phone number out of the group thread",
            "not share the private server link publicly",
        ],
        "discuss_budget": [
            "tell me if the ad spend is too high for this campaign",
            "let me know if the software subscription still fits the budget",
        ],
        "delay_until_ready": [
            "wait until the patch is live before announcing the feature",
            "hold the repost until the source is confirmed",
        ],
        "safety_check": [
            "tell me if the screenshot exposes personal details",
            "report it right away if the account gets suspicious login alerts",
        ],
        "give_feedback": [
            "tell me if the post tone feels too sharp",
            "let me know if the policy wording is confusing",
        ],
    },
    "sports": {
        "request_review": [
            "review the game plan before practice starts",
            "look over the travel roster before I submit it",
        ],
        "confirm_attendance": [
            "tell me if you can make the conditioning session",
            "confirm whether you can travel to the away game",
        ],
        "share_update": [
            "send the latest injury update before warmups",
            "post the final drill changes in the team chat",
        ],
        "follow_process": [
            "use the recovery checklist after training",
            "follow the physio plan before returning to full contact",
        ],
        "share_materials": [
            "bring the completed medical form tomorrow",
            "send the uniform sizes for the new players tonight",
        ],
        "raise_concern": [
            "tell me if the training load feels too heavy",
            "let me know if the travel plan seems unrealistic",
        ],
        "ask_for_help": [
            "help carry the water crates to the field",
            "help me check the cones before the session",
        ],
        "request_privacy": [
            "keep the lineup discussion inside the team",
            "not post the strategy board online",
        ],
        "discuss_budget": [
            "tell me if the travel cost is too high",
            "let me know if the new equipment order is too expensive",
        ],
        "delay_until_ready": [
            "wait for the physio assessment before full contact",
            "hold off on posting the roster until the coach confirms it",
        ],
        "safety_check": [
            "tell the trainer if the shoulder pain gets worse",
            "report it immediately if the turf comes loose again",
        ],
        "give_feedback": [
            "tell me if the drill explanation is unclear",
            "let me know if the team joke crossed a line",
        ],
    },
}


DOMAIN_PREFIXES = {
    "workplace": ["For the client project, ", "At work, "],
    "family": ["For the family plans, ", "At home, "],
    "friendship": ["For our plans with friends, ", "Between friends, "],
    "school": ["For class, ", "At school, "],
    "sales": ["For the buyer account, ", "In the sales process, "],
    "healthcare": ["For the clinic visit, ", "In the treatment plan, "],
    "community": ["For the neighborhood event, ", "In the community group, "],
    "housing": ["For the apartment, ", "In the building, "],
    "online": ["For the online channel, ", "On the platform, "],
    "sports": ["For the team, ", "At practice, "],
}


GENERIC_NEUTRAL_PREFIXES = [
    "",
    "Quick question: ",
    "Small request: ",
    "Just checking in: ",
    "Before I finalize this, ",
    "When you get a chance, ",
    "Hey, ",
    "Hi, ",
    "One quick thing: ",
]

GENERIC_MANIPULATIVE_PREFIXES = [
    "",
    "Honestly, ",
    "Listen, ",
    "Let us be real here: ",
    "Seriously, ",
    "I should not have to say this twice: ",
]

NEUTRAL_REASON_BRIDGES = [
    "so {reason}",
    "because {reason}",
    "since {reason}",
    "mainly because {reason}",
]

NEUTRAL_SECOND_SENTENCES = [
    "",
    "If the timing does not work, tell me and we can adjust.",
    "If anything is unclear, ask and I can explain it.",
    "No problem if something else is blocking it; I just want to plan around it.",
    "Let me know if you need anything from me first.",
    "If you see an issue, flag it before moving ahead.",
    "I would rather hear about a blocker early than chase it later.",
]

NEUTRAL_EDGE_SENTENCES = [
    "",
    "I know the timing is tight, but it is okay to say no if it will not work.",
    "I am under a bit of pressure on this too, so a direct answer would help.",
    "I would rather hear a no than get an unclear answer later.",
    "I know this is not ideal timing, so just tell me what is realistic.",
]

NEUTRAL_SIGNOFFS = [
    "",
    "Thanks.",
    "Appreciate it.",
    "Just keep me posted.",
    "Either way is fine; I only need clarity.",
]

NEUTRAL_BODY_TEMPLATES = [
    "Could you {action} {timeframe} {bridge}",
    "Please {action} {timeframe}; {reason_cap}",
    "Can you {action} {timeframe}? {reason_cap}",
    "I would like you to {action} {timeframe} {bridge}",
    "If possible, {action} {timeframe}; {reason_cap}",
    "Would you be able to {action} {timeframe}? {reason_cap}",
    "It would help if you could {action} {timeframe} {bridge}",
    "Just let me know if you can {action} {timeframe}; {reason_cap}",
]

TACTIC_SECOND_SENTENCES = {
    "guilt": [
        "It is frustrating when the rest of us end up carrying it instead.",
        "It is hard not to read your silence as a lack of support.",
        "Everyone else has already stretched to keep this moving.",
    ],
    "urgency": [
        "There is no point in waiting any longer on this.",
        "Every extra delay makes it harder to fix cleanly.",
        "The window to sort this out is closing fast.",
    ],
    "shame": [
        "At this stage, pushing back only makes you look careless.",
        "Most people would have handled it already by now.",
        "This should not still need to be explained.",
    ],
    "fear": [
        "If this blows up, people will remember where the delay happened.",
        "I do not want to see this come back on you later.",
        "The fallout will be much worse if you leave it hanging.",
    ],
    "reciprocity": [
        "I have shown up when you needed help, so this should not be hard.",
        "After everything I have done to back you up, this is a small ask.",
        "I would not be asking if I had not already carried my share for you.",
    ],
    "authority_pressure": [
        "This is not up for debate anymore.",
        "The decision has already been made, so stop reopening it.",
        "You do not need more discussion to do what I asked.",
    ],
    "social_proof": [
        "Everybody else has already aligned around it.",
        "You are the only person still acting like this is uncertain.",
        "The people who understand the situation are already on board.",
    ],
    "blame_shifting": [
        "If this slips, your choice here will be the obvious reason.",
        "From here on out, any delay will point back to this moment.",
        "Do not act surprised later if people connect the dots.",
    ],
    "emotional_blackmail": [
        "Your choice here tells me a lot about how much I matter to you.",
        "I should not have to question whether you actually support me.",
        "It is hard not to take this personally when you keep resisting.",
    ],
    "false_dichotomy": [
        "There is not really a neutral position on this.",
        "Your response will make it obvious where you stand.",
        "Either you back this or you are choosing not to.",
    ],
    "financial_pressure": [
        "Someone will end up paying for the waste if you do not move on it.",
        "The extra cost from waiting is not going to disappear on its own.",
        "If money gets burned here, your name will be attached to it.",
    ],
    "threat": [
        "If I have to escalate this, I will be specific about what happened.",
        "Do not force me to take this straight to the people above us.",
        "I am prepared to make this formal if you keep stalling.",
    ],
    "flattery_pressure": [
        "You are the one person people expect to handle this properly.",
        "I came to you because everyone knows you are the dependable one.",
        "If anyone can pull this together, it is you, which is why this is on you now.",
    ],
    "minimization": [
        "This is much simpler than you are making it.",
        "You are turning a straightforward ask into a whole issue.",
        "It really should not need this much extra discussion.",
    ],
    "isolation": [
        "Bringing in other people will only complicate it.",
        "Keep it between us unless you want unnecessary interference.",
        "Other voices will just muddy what should already be clear.",
    ],
}

MANIPULATIVE_CLOSERS = [
    "",
    "Do not make me chase this again.",
    "I need an answer, not another delay.",
    "You know exactly what I am asking.",
]

GENERIC_CONTEXT_TAILS = [
    "",
    "I am trying to avoid rework on the back end.",
    "I do not want this turning into a bigger issue later.",
    "It is easier to solve now than after other people get involved.",
]

SUBTLE_MANIPULATIVE_PHRASES = {
    "guilt": [
        "It would really help me if you did not leave me carrying this alone.",
        "I should not be the only one showing up for this.",
    ],
    "urgency": [
        "We are almost out of room to solve this cleanly.",
        "This is getting harder to contain the longer it sits.",
    ],
    "shame": [
        "At this point, most people would have dealt with it already.",
        "This is becoming hard to excuse.",
    ],
    "fear": [
        "I do not want this to become the thing that lands back on you later.",
        "This is the kind of delay people remember when things go wrong.",
    ],
    "reciprocity": [
        "Given how often I have covered for you, I expected less resistance here.",
        "After everything I have done on your side of things, this should be simple.",
    ],
    "authority_pressure": [
        "We are past the point where this needs your approval.",
        "The call has already been made, so just act on it.",
    ],
    "social_proof": [
        "No one else is treating this like an open question anymore.",
        "Everyone around this has already aligned except you.",
    ],
    "blame_shifting": [
        "If this slips, people are going to connect it to this moment.",
        "There will be a very obvious reason if this goes sideways now.",
    ],
    "emotional_blackmail": [
        "It would mean a lot if you stopped making me beg for basic support.",
        "I am learning a lot from how hard you are making this.",
    ],
    "false_dichotomy": [
        "There really is not a neutral version of this choice.",
        "Your response here is going to say which side of this you are on.",
    ],
    "financial_pressure": [
        "If cost goes up from here, there will be a very clear reason why.",
        "Waiting is going to turn into money lost for no reason.",
    ],
    "threat": [
        "If this keeps dragging, I will stop handling it informally.",
        "I am very close to escalating this directly.",
    ],
    "flattery_pressure": [
        "You are the only one I trust not to make a mess of it.",
        "People count on you for this kind of thing, which is why I came to you.",
    ],
    "minimization": [
        "This is not complicated enough to justify all this back-and-forth.",
        "You are treating a small ask like a major problem.",
    ],
    "isolation": [
        "The more people you pull into this, the messier it gets.",
        "Do not turn this into a group discussion unless you want it derailed.",
    ],
}

CONTRACTIONS = [
    (r"\bdo not\b", "don't"),
    (r"\bwill not\b", "won't"),
    (r"\bcannot\b", "can't"),
    (r"\bI am\b", "I'm"),
    (r"\bit is\b", "it's"),
    (r"\byou are\b", "you're"),
    (r"\bwe are\b", "we're"),
    (r"\bthat is\b", "that's"),
    (r"\bI would\b", "I'd"),
    (r"\bI have\b", "I've"),
]

CASUAL_REPLACEMENTS = [
    (r"\bPlease\b", "Pls"),
    (r"\bplease\b", "pls"),
    (r"\bAppreciate it\.", "thx."),
    (r"\bThanks\.", "thanks."),
    (r"\bJust checking in:", "just checking in -"),
    (r"\bQuick question:", "quick question -"),
]


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" ?", "?")
    return cleaned


def prepend(prefix: str, sentence: str) -> str:
    if not prefix:
        return sentence
    trimmed = sentence.lstrip()
    if not trimmed:
        return prefix.strip()

    stripped_prefix = prefix.rstrip()
    if stripped_prefix.endswith((",", ":")):
        if not trimmed.startswith("I "):
            trimmed = trimmed[0].lower() + trimmed[1:]
    return prefix + trimmed


def join_sentences(parts: list[str]) -> str:
    return clean_text(" ".join(part for part in parts if part))


def build_scenarios() -> list[dict[str, str]]:
    scenarios: list[dict[str, str]] = []
    for domain, intent_actions in DOMAIN_ACTIONS.items():
        for intent_label, actions in intent_actions.items():
            metadata = INTENT_METADATA[intent_label]
            for action in actions:
                scenarios.append(
                    {
                        "domain": domain,
                        "intent_label": intent_label,
                        "intent": metadata["intent"],
                        "action": action,
                    }
                )
    return scenarios


def shuffled_repeat(items: list, total: int, rng: random.Random) -> list:
    sequence = []
    while len(sequence) < total:
        batch = list(items)
        rng.shuffle(batch)
        sequence.extend(batch)
    return sequence[:total]


def choose_timeframe(intent_label: str, rng: random.Random) -> str:
    if intent_label == "delay_until_ready":
        return ""

    base = list(INTENT_METADATA[intent_label]["timeframes"])
    flexible = {
        "request_review": ["before I send the final version", "before it goes out"],
        "confirm_attendance": ["as soon as you know", "before I lock anything in"],
        "share_update": ["once you have the latest version", "before people start asking"],
        "follow_process": ["from here on out", "for this round"],
        "share_materials": ["when it is ready", "before the next step starts"],
        "raise_concern": ["while we still have room to change it", "before this gets locked in"],
        "ask_for_help": ["if you have a little bandwidth today", "when you have a minute"],
        "request_privacy": ["until we agree on what is public", "until it is okay to share"],
        "discuss_budget": ["before money is committed", "while we can still change course"],
        "delay_until_ready": ["for now", "a little longer", "for the moment", "before anything else changes"],
        "safety_check": ["as soon as you notice it", "right away if it changes"],
        "give_feedback": ["before I finalize it", "when you read through it"],
    }
    return rng.choice(base + flexible[intent_label])


def choose_domain_prefix(domain: str, rng: random.Random) -> str:
    options = DOMAIN_PREFIXES[domain] + ["", "", ""]
    return rng.choice(options)


def render_action(intent_label: str, action: str) -> str:
    if intent_label == "request_privacy" and action.startswith("not share "):
        remainder = action[len("not share ") :]
        return f"keep {remainder} private"
    return action


def apply_surface_variation(text: str, label: str, rng: random.Random) -> str:
    updated = text

    if rng.random() < 0.45:
        for pattern, target in CONTRACTIONS:
            if re.search(pattern, updated, flags=re.IGNORECASE) and rng.random() < 0.55:
                updated = re.sub(pattern, target, updated, count=1, flags=re.IGNORECASE)

    if rng.random() < 0.18:
        for pattern, target in CASUAL_REPLACEMENTS:
            if re.search(pattern, updated, flags=re.IGNORECASE):
                updated = re.sub(pattern, target, updated, count=1, flags=re.IGNORECASE)
                break

    if rng.random() < 0.12:
        updated = updated.replace("right away", "ASAP", 1)

    if rng.random() < 0.10:
        updated = updated.replace("because", "bc", 1)

    if rng.random() < 0.08 and updated.endswith("."):
        updated = updated[:-1]

    if rng.random() < 0.06:
        updated = updated.replace("...", ".", 1)
        updated = updated + (" ok?" if label == "manipulative" else " just let me know.")

    if rng.random() < 0.04:
        updated = updated.replace("you are", "ur", 1)

    return clean_text(updated)


def compose_neutral_text(scenario: dict[str, str], rng: random.Random) -> str:
    intent_label = scenario["intent_label"]
    reason = INTENT_METADATA[intent_label]["reason"]
    action = render_action(intent_label, scenario["action"])
    bridge = rng.choice(NEUTRAL_REASON_BRIDGES).format(reason=reason)
    body = rng.choice(NEUTRAL_BODY_TEMPLATES).format(
        action=action,
        timeframe=choose_timeframe(intent_label, rng),
        bridge=bridge,
        reason_cap=capitalize_first(reason),
    )
    body = prepend(choose_domain_prefix(scenario["domain"], rng), body)
    body = prepend(rng.choice(GENERIC_NEUTRAL_PREFIXES), body)

    second_sentence = rng.choice(NEUTRAL_SECOND_SENTENCES)
    if rng.random() < 0.35:
        second_sentence = join_sentences([second_sentence, rng.choice(GENERIC_CONTEXT_TAILS)])
    if rng.random() < 0.22:
        second_sentence = join_sentences([second_sentence, rng.choice(NEUTRAL_EDGE_SENTENCES)])

    text = join_sentences([body, second_sentence, rng.choice(NEUTRAL_SIGNOFFS)])
    return apply_surface_variation(text, "not_manipulative", rng)


def compose_manipulative_text(scenario: dict[str, str], tactic: str, rng: random.Random) -> str:
    intent_label = scenario["intent_label"]
    template = rng.choice(MANIPULATIVE_TEMPLATES[tactic])
    action = render_action(intent_label, scenario["action"])
    body = template.format(action=action, timeframe=choose_timeframe(intent_label, rng))
    body = prepend(choose_domain_prefix(scenario["domain"], rng), body)
    body = prepend(rng.choice(GENERIC_MANIPULATIVE_PREFIXES), body)

    follow_up = ""
    if rng.random() < 0.85:
        follow_up = rng.choice(TACTIC_SECOND_SENTENCES[tactic])
    if rng.random() < 0.30:
        follow_up = join_sentences([follow_up, rng.choice(SUBTLE_MANIPULATIVE_PHRASES[tactic])])
    if rng.random() < 0.25:
        follow_up = join_sentences([follow_up, rng.choice(MANIPULATIVE_CLOSERS)])

    text = join_sentences([body, follow_up])
    return apply_surface_variation(text, "manipulative", rng)


def build_rows(total_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> list[dict[str, str | int]]:
    if total_rows < 2:
        raise ValueError("total_rows must be at least 2")

    rng = random.Random(seed)
    scenarios = build_scenarios()
    target_neutral = total_rows // 2
    target_manipulative = total_rows - target_neutral

    neutral_scenarios = shuffled_repeat(scenarios, target_neutral, rng)
    manipulative_scenarios = shuffled_repeat(scenarios, target_manipulative, rng)
    manipulative_tactics = shuffled_repeat(TACTICS, target_manipulative, rng)

    rows: list[dict[str, str | int]] = []
    seen_texts: set[str] = set()
    row_id = 1

    def add_row(
        *,
        label: str,
        scenario: dict[str, str],
        tactic: str,
        severity: str,
        row_seed_base: int,
    ) -> None:
        nonlocal row_id
        for attempt in range(200):
            local_rng = random.Random(row_seed_base + attempt * 9973)
            if label == "not_manipulative":
                text = compose_neutral_text(scenario, local_rng)
            else:
                text = compose_manipulative_text(scenario, tactic, local_rng)

            if text not in seen_texts:
                seen_texts.add(text)
                rows.append(
                    {
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "intent_label": scenario["intent_label"],
                        "intent": scenario["intent"],
                        "manipulation_type": tactic,
                        "domain": scenario["domain"],
                        "severity": severity,
                    }
                )
                row_id += 1
                return

        raise RuntimeError("Could not generate a unique text after many attempts")

    for index, scenario in enumerate(neutral_scenarios):
        add_row(
            label="not_manipulative",
            scenario=scenario,
            tactic="none",
            severity="none",
            row_seed_base=seed * 10_000 + index,
        )

    for index, (scenario, tactic) in enumerate(zip(manipulative_scenarios, manipulative_tactics, strict=False)):
        add_row(
            label="manipulative",
            scenario=scenario,
            tactic=tactic,
            severity=TACTIC_SEVERITY[tactic],
            row_seed_base=seed * 100_000 + index,
        )

    rng.shuffle(rows)
    for row_id, row in enumerate(rows, start=1):
        row["id"] = row_id

    return rows


def write_rows(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "text",
                "label",
                "intent_label",
                "intent",
                "manipulation_type",
                "domain",
                "severity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible generation.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path.")
    args = parser.parse_args()

    rows = build_rows(total_rows=args.rows, seed=args.seed)
    write_rows(rows, args.output)

    print(f"rows={len(rows)}")
    print(f"seed={args.seed}")
    print(f"output_path={args.output}")


if __name__ == "__main__":
    main()
