/*
 * Exhale Project Original (2026)
 * ozyern (github.com/ozyern)
 * Licensed Under GPL-3.0 | see git history for contributors
 */

package com.ozyern.exhale.ui.component

import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.RuntimeShader
import android.graphics.Shader
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ShaderBrush
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import coil3.imageLoader
import coil3.request.ImageRequest
import coil3.request.allowHardware
import coil3.toBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/** The cover is shrunk to this many pixels a side: the field is soft gradients, and this is all it samples. */
private const val Side = 48
private const val GridW = 48
private const val GridH = 84

private const val FluidAgsl = """
uniform shader cover;
uniform shader previous;
uniform float2 size;
uniform float time;
uniform float blend;
uniform float brightness;

half3 field(float2 p, float time, float aspect, bool useCover) {
    half3 acc = half3(0.0);
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float dir = (mod(fi, 2.0) < 0.5) ? 1.0 : -1.0;
        float a = time * (0.045 + 0.02 * fi) * dir + fi * 1.9;
        float c = cos(a);
        float s = sin(a);
        float scale = 1.0 / (1.35 + 0.45 * fi);
        float2 q = float2(c * p.x - s * p.y, s * p.x + c * p.y) * scale;
        float u = q.x + 0.10 * sin(time * 0.17 + fi * 2.3 + q.y * 4.0) + 0.012 * sin(time * 0.31 + fi + q.y * 4.5) + 0.5 + sin(time * 0.05 + fi) * 0.18;
        float v = q.y + 0.10 * cos(time * 0.13 + fi * 1.1 + q.x * 4.0) + 0.012 * cos(time * 0.27 + fi + q.x * 4.5) + 0.5 + cos(time * 0.04 + fi * 1.7) * 0.18;
        float2 uv = clamp(float2(u, v), 0.02, 0.98) * $Side.0;
        half w = half(1.9 / (1.0 + 1.5 * fi));
        acc += (useCover ? cover.eval(uv).rgb : previous.eval(uv).rgb) * w;
    }
    return acc / 4.27;
}

half4 main(float2 fragCoord) {
    float aspect = size.x / size.y;
    float2 p = (fragCoord / size - 0.5) * float2(aspect, 1.0);
    half3 col = mix(field(p, time, aspect, false), field(p, time, aspect, true), half(blend));

    half lift = half(0.0);
    for (int j = 0; j < 3; j++) {
        float fj = float(j);
        float2 centre = (float2(0.5 + 0.34 * sin(time * 0.11 + fj * 2.1), 0.5 + 0.30 * cos(time * 0.09 + fj * 1.7)) - 0.5) * float2(aspect, 1.0);
        float d = length(p - centre);
        lift += half(exp(-d * d * 5.0) * (0.55 + 0.45 * sin(time * 0.4 + fj * 1.3)));
    }
    col *= half(1.0) + lift * half(0.35);

    half luma = dot(col, half3(0.299, 0.587, 0.114));
    col = max(half3(luma) + (col - half3(luma)) * 1.55, half3(0.0));
    col = clamp(col * col * (half3(3.0) - half3(2.0) * col), half3(0.0), half3(1.4));

    float vignette = 1.0 - smoothstep(0.2, 1.3, length(p));
    col *= half(brightness * (0.60 + 0.40 * vignette));

    float n = fract(sin(dot(fragCoord, float2(12.9898, 78.233))) * 43758.5453);
    col += half((n - 0.5) * 0.012);
    return half4(col, 1.0);
}
"""

/** A cover and the texture the shader reads it through, made once rather than every frame. */
private class Cover(bitmap: Bitmap) {
    val shader: BitmapShader = BitmapShader(bitmap, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
    val pixels = IntArray(Side * Side).also {
        bitmap.getPixels(it, 0, Side, 0, 0, Side, Side)
    }
}

private fun evalFieldCpu(px: Float, py: Float, time: Float, pixels: IntArray, outRgb: FloatArray) {
    var rAcc = 0f
    var gAcc = 0f
    var bAcc = 0f
    for (i in 0 until 5) {
        val fi = i.toFloat()
        val dir = if (i % 2 == 0) 1f else -1f
        val a = time * (0.045f + 0.02f * fi) * dir + fi * 1.9f
        val c = cos(a)
        val s = sin(a)
        val scale = 1f / (1.35f + 0.45f * fi)
        val qx = (c * px - s * py) * scale
        val qy = (s * px + c * py) * scale
        val u = qx + 0.10f * sin(time * 0.17f + fi * 2.3f + qy * 4.0f) + 0.012f * sin(time * 0.31f + fi + qy * 4.5f) + 0.5f + sin(time * 0.05f + fi) * 0.18f
        val v = qy + 0.10f * cos(time * 0.13f + fi * 1.1f + qx * 4.0f) + 0.012f * cos(time * 0.27f + fi + qx * 4.5f) + 0.5f + cos(time * 0.04f + fi * 1.7f) * 0.18f
        val x = (u.coerceIn(0.02f, 0.98f) * (Side - 1)).toInt()
        val y = (v.coerceIn(0.02f, 0.98f) * (Side - 1)).toInt()
        val color = pixels[y * Side + x]
        val w = 1.9f / (1.0f + 1.5f * fi)
        rAcc += ((color shr 16) and 0xFF) * (1f / 255f) * w
        gAcc += ((color shr 8) and 0xFF) * (1f / 255f) * w
        bAcc += (color and 0xFF) * (1f / 255f) * w
    }
    outRgb[0] = rAcc / 4.27f
    outRgb[1] = gAcc / 4.27f
    outRgb[2] = bAcc / 4.27f
}

private fun renderFluidBitmap(
    outPixels: IntArray,
    nowPixels: IntArray,
    prevPixels: IntArray,
    aspect: Float,
    time: Float,
    blend: Float,
    brightness: Float
) {
    val rgbNow = FloatArray(3)
    val rgbPrev = FloatArray(3)
    val cx = FloatArray(3)
    val cy = FloatArray(3)
    val cAmp = FloatArray(3)
    for (j in 0 until 3) {
        val fj = j.toFloat()
        cx[j] = (0.5f + 0.34f * sin(time * 0.11f + fj * 2.1f) - 0.5f) * aspect
        cy[j] = 0.5f + 0.30f * cos(time * 0.09f + fj * 1.7f) - 0.5f
        cAmp[j] = 0.55f + 0.45f * sin(time * 0.4f + fj * 1.3f)
    }

    var idx = 0
    for (gy in 0 until GridH) {
        val py = (gy + 0.5f) / GridH - 0.5f
        for (gx in 0 until GridW) {
            val px = ((gx + 0.5f) / GridW - 0.5f) * aspect
            evalFieldCpu(px, py, time, nowPixels, rgbNow)
            var r = rgbNow[0]
            var g = rgbNow[1]
            var b = rgbNow[2]
            if (blend < 0.999f) {
                evalFieldCpu(px, py, time, prevPixels, rgbPrev)
                r = rgbPrev[0] + (r - rgbPrev[0]) * blend
                g = rgbPrev[1] + (g - rgbPrev[1]) * blend
                b = rgbPrev[2] + (b - rgbPrev[2]) * blend
            }

            var lift = 0f
            for (j in 0 until 3) {
                val dx = px - cx[j]
                val dy = py - cy[j]
                val d2 = dx * dx + dy * dy
                lift += exp(-d2 * 5.0f) * cAmp[j]
            }
            val liftMul = 1.0f + lift * 0.35f
            r *= liftMul
            g *= liftMul
            b *= liftMul

            val luma = 0.299f * r + 0.587f * g + 0.114f * b
            r = (luma + (r - luma) * 1.55f).coerceAtLeast(0f)
            g = (luma + (g - luma) * 1.55f).coerceAtLeast(0f)
            b = (luma + (b - luma) * 1.55f).coerceAtLeast(0f)

            r = (r * r * (3.0f - 2.0f * r)).coerceIn(0f, 1.4f)
            g = (g * g * (3.0f - 2.0f * g)).coerceIn(0f, 1.4f)
            b = (b * b * (3.0f - 2.0f * b)).coerceIn(0f, 1.4f)

            val lenP = sqrt(px * px + py * py)
            val tVig = ((lenP - 0.2f) / (1.3f - 0.2f)).coerceIn(0f, 1f)
            val vignette = 1.0f - tVig * tVig * (3.0f - 2.0f * tVig)
            val vMul = brightness * (0.60f + 0.40f * vignette)

            val ir = (r * vMul * 255f).toInt().coerceIn(0, 255)
            val ig = (g * vMul * 255f).toInt().coerceIn(0, 255)
            val ib = (b * vMul * 255f).toInt().coerceIn(0, 255)
            outPixels[idx++] = (0xFF shl 24) or (ir shl 16) or (ig shl 8) or ib
        }
    }
}

@Composable
fun FluidArtworkBackground(
    url: String?,
    modifier: Modifier = Modifier,
    brightness: Float = 0.9f,
    paused: Boolean,
) {
    val context = LocalContext.current
    var current by remember { mutableStateOf<Cover?>(null) }
    var previous by remember { mutableStateOf<Cover?>(null) }
    val blend = remember { Animatable(1f) }
    val clock = remember { mutableFloatStateOf(0f) }
    val isPaused by rememberUpdatedState(paused)

    LaunchedEffect(url) {
        if (url == null) return@LaunchedEffect
        val request = ImageRequest.Builder(context).data(url).size(Side, Side).allowHardware(false).build()
        val bitmap = runCatching { withContext(Dispatchers.IO) { context.imageLoader.execute(request).image?.toBitmap() } }
            .getOrNull() ?: return@LaunchedEffect
        val small = if (bitmap.width == Side && bitmap.height == Side) bitmap else Bitmap.createScaledBitmap(bitmap, Side, Side, true)
        previous = current ?: Cover(small)
        current = Cover(small)
        blend.snapTo(0f)
        blend.animateTo(1f, tween(1_400, easing = LinearOutSlowInEasing))
    }

    LaunchedEffect(Unit) {
        var last = 0L
        while (true) {
            if (isPaused || !com.ozyern.exhale.utils.isAppVisible) {
                snapshotFlow { isPaused }.first { !it }
                com.ozyern.exhale.utils.awaitAppVisible()
                last = 0L
            }
            androidx.compose.runtime.withFrameNanos { now ->
                if (last != 0L) clock.floatValue += ((now - last) / 1_000_000_000f).coerceAtMost(0.1f)
                last = now
            }
            com.ozyern.exhale.utils.awaitAmbientStep()
        }
    }

    val shader = remember { runCatching { RuntimeShader(FluidAgsl) }.getOrNull() }
    val brush = remember(shader) { shader?.let { ShaderBrush(it) } }
    val fallbackBitmap = remember { Bitmap.createBitmap(GridW, GridH, Bitmap.Config.ARGB_8888) }
    val fallbackPixels = remember { IntArray(GridW * GridH) }
    val fallbackPaint = remember { Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG) }
    val dstRect = remember { RectF() }

    Box(
        modifier
            .fillMaxSize()
            .background(Color.Black)
            .drawBehind {
                val now = current ?: return@drawBehind
                val prev = previous ?: now
                if (shader != null && brush != null) {
                    shader.setInputShader("cover", now.shader)
                    shader.setInputShader("previous", prev.shader)
                    shader.setFloatUniform("size", size.width, size.height)
                    shader.setFloatUniform("time", clock.floatValue)
                    shader.setFloatUniform("blend", blend.value)
                    shader.setFloatUniform("brightness", brightness)
                    drawRect(brush)
                } else if (size.width > 0f && size.height > 0f) {
                    renderFluidBitmap(
                        outPixels = fallbackPixels,
                        nowPixels = now.pixels,
                        prevPixels = prev.pixels,
                        aspect = size.width / size.height,
                        time = clock.floatValue,
                        blend = blend.value,
                        brightness = brightness
                    )
                    fallbackBitmap.setPixels(fallbackPixels, 0, GridW, 0, 0, GridW, GridH)
                    dstRect.set(0f, 0f, size.width, size.height)
                    drawIntoCanvas { canvas ->
                        canvas.nativeCanvas.drawBitmap(fallbackBitmap, null, dstRect, fallbackPaint)
                    }
                }
            },
    )
}
