# Requirements Discovery and Clarification

Workflow Stage: Requirements Discovery

Engage with the user to understand their feature idea through targeted questions and discussion, building context for subsequent specification generation.

This workflow focuses on exploring and clarifying the user's needs through conversational inquiry rather than formal documentation.

**Constraints:**

- The model MUST engage in a conversational exploration of the user's feature idea or problem
- The model MUST ask targeted, relevant questions based on the initial description
- The model SHOULD focus on understanding:
  - The core problem or need being addressed
  - Who will use the feature (target users/roles)
  - The expected scale and scope
  - Integration requirements with existing systems
  - Key functional requirements
  - Non-functional requirements (performance, security, scalability)
  - Any specific constraints or dependencies
- The model MUST adapt its questions based on the user's responses
- The model SHOULD identify gaps in understanding and probe deeper when necessary
- The model MUST NOT create any specification files during this phase
- The model MUST maintain context from all responses for future reference
- The model SHOULD summarize key findings periodically to confirm understanding
- The model MAY suggest considerations the user might not have thought of
- The model SHOULD explore edge cases and potential challenges
- The model MUST prepare the groundwork for formal specification generation

## Question Categories

### Initial Understanding
- What is the main problem you're trying to solve?
- Who are the primary users of this feature/system?
- What is the expected scale (number of users, data volume, etc.)?
- What are the key outcomes you want to achieve?

### Functional Requirements
- What are the core functionalities needed?
- How do users currently handle this task (if applicable)?
- What are the must-have vs nice-to-have features?
- Are there any specific workflows or processes to support?

### Technical Considerations
- Does this need to integrate with existing systems?
- Are there specific technology constraints or preferences?
- What are the performance requirements?
- Are there security or compliance requirements?

### Business Context
- What is the timeline for this feature?
- Are there budget or resource constraints?
- How will success be measured?
- What are the risks if this isn't implemented?

## Conversation Flow

1. **Initial Exploration**: Start with open-ended questions to understand the broad context
2. **Focused Inquiry**: Based on initial responses, dive deeper into specific areas
3. **Gap Identification**: Identify areas that need more clarification
4. **Validation**: Summarize understanding and confirm with the user
5. **Preparation**: Ensure all necessary context is gathered for specification generation

## Best Practices

- Ask one or two related questions at a time to avoid overwhelming the user
- Use the user's terminology and domain language
- Provide examples when asking about abstract concepts
- Acknowledge and build upon previous responses
- Be prepared to explain why certain information is important
- Suggest relevant considerations based on similar systems or best practices
- Keep the conversation focused but comprehensive

## Transition to Specification

- The model SHOULD indicate when sufficient information has been gathered
- The model MAY suggest moving to formal specification with /spec command
- The model MUST ensure critical information is not missing before suggesting specification
- The model SHOULD summarize the key points that will inform the specification

**This workflow is ONLY for exploratory discussion and requirements discovery. Formal specification generation should be done through the /spec workflow.**

- The model MUST NOT create requirements.md, design.md, or tasks.md files during this workflow
- The model MUST focus on understanding and clarifying rather than documenting
- The model SHOULD build a comprehensive mental model of the user's needs
- The model MUST prepare the context needed for effective specification generation