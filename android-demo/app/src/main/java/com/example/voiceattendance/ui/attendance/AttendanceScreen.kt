package com.example.voiceattendance.ui.attendance

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.voiceattendance.data.model.SessionInfo

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AttendanceScreen(
    viewModel: AttendanceViewModel,
    onNavigateBack: () -> Unit
) {
    val state by viewModel.state.collectAsState()
    val secondsRemaining by viewModel.challengeSecondsRemaining.collectAsState()

    LaunchedEffect(Unit) {
        if (state is AttendanceState.Idle) {
            viewModel.loadHome()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    val title = when (state) {
                        is AttendanceState.HomeLoaded,
                        is AttendanceState.LoadingHome,
                        is AttendanceState.Idle -> "Today's Classes"
                        is AttendanceState.ReadyToRecord,
                        is AttendanceState.Recording,
                        is AttendanceState.Uploading -> "Marking Attendance"
                        is AttendanceState.Result -> "Result"
                        is AttendanceState.Error -> "Error"
                    }
                    Text(title)
                },
                navigationIcon = {
                    IconButton(onClick = {
                        if (state is AttendanceState.Recording || state is AttendanceState.Uploading) {
                            // Don't allow back during recording
                        } else {
                            onNavigateBack()
                        }
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            contentAlignment = Alignment.TopCenter
        ) {
            when (val s = state) {
                is AttendanceState.Idle,
                is AttendanceState.LoadingHome -> LoadingContent()

                is AttendanceState.HomeLoaded -> ClassListContent(
                    student = s.student.displayName,
                    tenantName = s.student.tenantName,
                    sessions = s.sessions,
                    onMarkSession = { session -> viewModel.selectSession(session) }
                )

                is AttendanceState.ReadyToRecord -> ReadyToRecordContent(
                    session = s.session,
                    challengePhrase = viewModel.challengePhrase,
                    secondsRemaining = secondsRemaining,
                    onStart = { viewModel.startRecording(s.session) }
                )

                is AttendanceState.Recording -> ActiveRecordingContent(
                    session = s.session,
                    phase = "Recording... speak clearly",
                    challengePhrase = viewModel.challengePhrase,
                    secondsRemaining = secondsRemaining
                )

                is AttendanceState.Uploading -> ActiveRecordingContent(
                    session = s.session,
                    phase = "Verifying your voice...",
                    challengePhrase = viewModel.challengePhrase,
                    secondsRemaining = secondsRemaining
                )

                is AttendanceState.Result -> ResultContent(
                    session   = s.session,
                    isPresent = s.isPresent,
                    reason    = s.reason,
                    onMarkAnother = { viewModel.returnToHome() },
                    onDone        = onNavigateBack
                )

                is AttendanceState.Error -> ErrorContent(
                    message = s.message,
                    onRetry = { viewModel.loadHome() }
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Loading
// ─────────────────────────────────────────────────────────────
@Composable
private fun LoadingContent() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(16.dp))
        Text("Loading classes...", color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ─────────────────────────────────────────────────────────────
//  Today's class list
// ─────────────────────────────────────────────────────────────
@Composable
private fun ClassListContent(
    student: String,
    tenantName: String,
    sessions: List<SessionInfo>,
    onMarkSession: (SessionInfo) -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(vertical = 16.dp)
    ) {
        item {
            // Student greeting header
            Text(
                text = "Good day,",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = student,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = tenantName,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(20.dp))
            Text(
                text = "Today's Schedule",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(modifier = Modifier.height(8.dp))
        }

        if (sessions.isEmpty()) {
            item {
                Box(modifier = Modifier.fillParentMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No classes scheduled for today", textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        } else {
            items(sessions, key = { it.sessionId }) { session ->
                ClassCard(session = session, onMark = { onMarkSession(session) })
            }
        }
    }
}

@Composable
private fun ClassCard(session: SessionInfo, onMark: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = session.className,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = session.timeRange,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            when (session.attendanceStatus) {
                "PRESENT" -> {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null,
                            tint = Color(0xFF4CAF50), modifier = Modifier.size(18.dp))
                        Text("Present", color = Color(0xFF4CAF50), fontWeight = FontWeight.SemiBold,
                            fontSize = 13.sp)
                    }
                }
                "NOT_MARKED" -> {
                    Button(
                        onClick = onMark,
                        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                        modifier = Modifier.height(34.dp)
                    ) {
                        Text("Mark", fontSize = 13.sp)
                    }
                }
                "UPCOMING" -> {
                    Surface(shape = RoundedCornerShape(6.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant) {
                        Text("Upcoming", modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                "ENDED" -> {
                    Surface(shape = RoundedCornerShape(6.dp),
                        color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.4f)) {
                        Text("Ended", modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                else -> {}
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Recording in progress
// ─────────────────────────────────────────────────────────────
@Composable
private fun ReadyToRecordContent(session: SessionInfo, challengePhrase: String?, secondsRemaining: Int = 10, onStart: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Marking attendance for", style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer)
                Text(session.className, style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
            }
        }
        
        if (challengePhrase != null) {
            Spacer(modifier = Modifier.height(24.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (secondsRemaining <= 3) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Time Remaining:",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = "${secondsRemaining}s",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { (secondsRemaining / 10f).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().height(6.dp),
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                    Spacer(modifier = Modifier.height(14.dp))
                    Text("Please say the following phrase:", 
                        style = MaterialTheme.typography.labelMedium,
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "\"$challengePhrase\"",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(48.dp))
        Button(
            onClick = onStart,
            modifier = Modifier.fillMaxWidth().height(56.dp)
        ) {
            Text("Start Recording", fontSize = 18.sp)
        }
    }
}

@Composable
private fun ActiveRecordingContent(session: SessionInfo, phase: String, challengePhrase: String? = null, secondsRemaining: Int = 10) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Marking attendance for", style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer)
                Text(session.className, style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onPrimaryContainer)
                Text(session.timeRange, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer)
            }
        }
        
        if (challengePhrase != null) {
            Spacer(modifier = Modifier.height(24.dp))
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = if (secondsRemaining <= 3) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Time Remaining:",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = "${secondsRemaining}s",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { (secondsRemaining / 10f).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().height(6.dp),
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                    Spacer(modifier = Modifier.height(14.dp))
                    Text("Please say the following phrase:", 
                        style = MaterialTheme.typography.labelMedium,
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "\"$challengePhrase\"",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = if (secondsRemaining <= 3) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
        
        Spacer(modifier = Modifier.height(40.dp))
        CircularProgressIndicator(modifier = Modifier.size(64.dp))
        Spacer(modifier = Modifier.height(20.dp))
        Text(phase, style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ─────────────────────────────────────────────────────────────
//  Result
// ─────────────────────────────────────────────────────────────
@Composable
private fun ResultContent(
    session: SessionInfo,
    isPresent: Boolean,
    reason: String,
    onMarkAnother: () -> Unit,
    onDone: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Session context
        Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(session.className, style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold)
                Text(session.timeRange, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Spacer(modifier = Modifier.height(32.dp))

        if (isPresent) {
            Icon(Icons.Default.CheckCircle, contentDescription = null,
                tint = Color(0xFF4CAF50), modifier = Modifier.size(72.dp))
            Spacer(modifier = Modifier.height(12.dp))
            Text("Attendance Marked", style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold, color = Color(0xFF4CAF50))
            Text("You are marked Present", style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            Icon(Icons.Default.Warning, contentDescription = null,
                tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(72.dp))
            Spacer(modifier = Modifier.height(12.dp))
            val title = when {
                reason.contains("Already marked", ignoreCase = true) -> "Already Marked"
                else -> "Not Marked"
            }
            Text(title, style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.error)
            Text(reason, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
        }
        Spacer(modifier = Modifier.height(40.dp))
        Button(onClick = onMarkAnother, modifier = Modifier.fillMaxWidth().height(52.dp)) {
            Text("Back to Classes")
        }
        Spacer(modifier = Modifier.height(10.dp))
        OutlinedButton(onClick = onDone, modifier = Modifier.fillMaxWidth()) {
            Text("Home")
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Error
// ─────────────────────────────────────────────────────────────
@Composable
private fun ErrorContent(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(Icons.Default.Warning, contentDescription = null,
            tint = MaterialTheme.colorScheme.error, modifier = Modifier.size(64.dp))
        Spacer(modifier = Modifier.height(16.dp))
        Text("Something went wrong", style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(8.dp))
        Text(message, style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(32.dp))
        Button(onClick = onRetry, modifier = Modifier.fillMaxWidth()) { Text("Try Again") }
    }
}
