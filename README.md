{{- if .Values.externalSecret.enabled }}
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: {{ .Values.externalSecret.name }}
  namespace: {{ .Values.namespace }}
spec:
  refreshInterval: {{ .Values.externalSecret.refreshInterval }}
  secretStoreRef:
    name: {{ .Values.externalSecret.secretStoreRef.name }}
    kind: {{ .Values.externalSecret.secretStoreRef.kind }}
  target:
    name: {{ .Values.externalSecret.name }}
    creationPolicy: Owner
  data:
  - secretKey: connection_string
    remoteRef:
      key: {{ .Values.externalSecret.remoteSecretKey }}
---
{{- end }}
apiVersion: batch/v1
kind: CronJob
metadata:
  name: {{ include "litellm-pg2bq-sync.fullname" . }}
  namespace: {{ .Values.namespace }}
  labels:
    {{- include "litellm-pg2bq-sync.labels" . | nindent 4 }}
spec:
  schedule: {{ .Values.cronjob.schedule | quote }}
  timeZone: {{ .Values.cronjob.timeZone | quote }}
  successfulJobsHistoryLimit: {{ .Values.cronjob.successfulJobsHistoryLimit }}
  failedJobsHistoryLimit: {{ .Values.cronjob.failedJobsHistoryLimit }}
  concurrencyPolicy: {{ .Values.cronjob.concurrencyPolicy }}
  jobTemplate:
    spec:
      backoffLimit: {{ .Values.cronjob.backoffLimit }}
      template:
        metadata:
          labels:
            {{- include "litellm-pg2bq-sync.selectorLabels" . | nindent 12 }}
        spec:
          restartPolicy: OnFailure
          serviceAccount: {{ .Values.serviceAccount.name }}
          containers:
          - name: sync
            image: "{{ .Values.image.registry }}/{{ .Values.image.repository }}:{{ .Values.image.tag }}"
            imagePullPolicy: {{ .Values.image.pullPolicy }}
            resources:
              {{- toYaml .Values.resources | nindent 14 }}
            env:
            - name: TZ
              value: "Asia/Kolkata"
            - name: POSTGRES_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.externalSecret.name }}
                  key: connection_string
            volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
          volumes:
          - name: config
            configMap:
              name: {{ include "litellm-pg2bq-sync.fullname" . }}-config
