const predictedTeamElement = document.getElementById('predicted-team');
const originalInputElement = document.getElementById('original-input');

const originalInput = {
    city: 'Mumbai',
    home: 'Mumbai Indians',
    away: 'Delhi Capitals',
    toss_winner: 'Mumbai Indians',
    toss_decision: 'bat',
    venue: 'Wankhede Stadium'
};

const predictedTeam = 'Mumbai Indians';

predictedTeamElement.textContent = predictedTeam;

const originalInputHtml = '<ul>';
for (const [key, value] of Object.entries(originalInput)) {
    originalInputHtml += `<li><strong>${key}:</strong> ${value}</li>`;
}
originalInputHtml += '</ul>';

originalInputElement.innerHTML = originalInputHtml;