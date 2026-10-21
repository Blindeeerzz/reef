# [Security Tracking] Checklist de hardening para integración de REEF en ecosistema WillMax

**Autor:** Aegis (Director de Ciberseguridad, WillMax AI Systems)
**Fecha:** 21 de octubre de 2026
**Estado:** EN EVALUACIÓN — ningún punto aprobado para producción todavía

## Contexto

Fork de evaluación de `Human-Agent-Society/reef` para dos casos de uso potenciales en WillMax AI Systems:

1. **Hephaestus Forge** (futuro): usar la infraestructura de REEF (loop Serve→Observe→Grow→Commit) para que Hephaestus pueda generar y supervisar subagentes nuevos capaces de auto-mejorarse.
2. **Auto-mejora de agentes del consorcio** (Aegis, Chloe, Jasmin, Lili, Pere, Edu, Lilith): incorporar un mecanismo de evolución de Skills/procedimientos en nuestros propios System Prompts, SIN alterar la personalidad/identidad base de cada agente.

Ninguno de los dos casos está aprobado para producción todavía. Este documento es el checklist de aceptación de seguridad obligatorio antes de cualquier despliegue real.

## Hallazgos de la auditoría inicial (Aegis, 21-oct-2026)

- El repo oficial YA resolvió su propio riesgo de sandboxing (issue #18 upstream, cerrado y verificado): existe `reef/sandbox/` con soporte E2B (microVMs), aislamiento de filesystem, sin credenciales de host heredadas.
- Punto de atención: modo "Reefine" (evolución de harness) documenta explícitamente que sin sandbox configurado (`REEF_PROPOSER_SANDBOX=none`), el proceso corre sin aislamiento — solo usar en máquina de confianza, nunca en producción.

## Checklist de aceptación OBLIGATORIO antes de integrar

### Para el caso "Hephaestus Forge" (subagentes nuevos)
- [ ] Executor E2B (o equivalente microVM/contenedor) forzado por defecto en cualquier despliegue WillMax — nunca el executor local/subprocess en producción.
- [ ] Política de red deny-by-default con allowlist explícita (solo endpoints de modelo autorizados por WillMax).
- [ ] Cero credenciales WillMax (AGENT_TOKEN_*, Docker socket, SSH, cloud creds) accesibles desde el entorno sandboxed. Ref. incidente Hecate (02-sept-2026): socket Docker RW expuesto = container breakout real.
- [ ] Límites de CPU/memoria/PID/tiempo de pared por episodio, con terminación garantizada del árbol de procesos.
- [ ] Todo candidato "Commit" (nueva versión publicada) pasa por revisión humana o de Aegis antes de ir a producción. Cero auto-despliegue sin aprobación.

### Para el caso "Auto-mejora de agentes del consorcio" (riesgo superior)
- [ ] Separación arquitectónica estricta: núcleo de identidad/personalidad/tono de cada agente = INMUTABLE. Solo Skills/procedimientos/conocimiento operativo son superficie evolutiva.
- [ ] Human-in-the-loop obligatorio: ninguna actualización de prompt/skill se despliega a un agente en producción sin aprobación explícita de Abel o Lilith.
- [ ] Versionado completo + rollback instantáneo si una actualización degrada comportamiento (detectado por feedback negativo, quejas, o tests de regresión de personalidad).
- [ ] Entorno de staging aislado para probar candidatos de mejora antes de tocar al agente real; el agente de pruebas NUNCA tiene acceso a herramientas de producción (CRM, correo, Docker, repos reales).
- [ ] Tests de regresión de personalidad/tono antes de cada merge (verificar que el agente sigue "sonando" como él mismo).

## Siguiente paso

Evaluación de implementación: decisión pendiente sobre si el desarrollo se hace vía Antigravity o se migra a OpenCode (por coste de tokens). Aegis recomendará modelo compatible con razonamiento + código, priorizando opciones gratuitas/económicas pero competentes.

## Responsables

- Auditoría de seguridad y checklist: Aegis
- Decisión estratégica: Lilith (CEO) + Abel
- Implementación de infraestructura: Antigravity (o vía OpenCode, pendiente de decidir)
