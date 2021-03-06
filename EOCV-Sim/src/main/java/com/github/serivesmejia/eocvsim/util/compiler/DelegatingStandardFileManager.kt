/*
 * Copyright (c) 2021 Sebastian Erives & (c) 2017 Robert Atkinson
 *
 * Based from the FTC SDK's org.firstinspires.ftc.onbotjava.OnBotJavaDelegatingStandardFileManager
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package com.github.serivesmejia.eocvsim.util.compiler

import java.io.File
import javax.tools.ForwardingJavaFileManager
import javax.tools.JavaFileManager
import javax.tools.JavaFileObject
import javax.tools.StandardJavaFileManager

open class DelegatingStandardFileManager(
    val delegate: StandardJavaFileManager
) : ForwardingJavaFileManager<StandardJavaFileManager>(delegate), StandardJavaFileManager {

    override fun getJavaFileObjectsFromFiles(files: MutableIterable<File>): MutableIterable<JavaFileObject> =
        delegate.getJavaFileObjectsFromFiles(files)

    override fun getJavaFileObjects(vararg files: File): MutableIterable<JavaFileObject> =
        delegate.getJavaFileObjects(*files)

    override fun getJavaFileObjects(vararg names: String): MutableIterable<JavaFileObject> =
        delegate.getJavaFileObjects(*names)

    override fun getJavaFileObjectsFromStrings(names: MutableIterable<String>): MutableIterable<JavaFileObject> =
        delegate.getJavaFileObjectsFromStrings(names)

    override fun setLocation(location: JavaFileManager.Location, files: MutableIterable<File>) =
        delegate.setLocation(location, files)

    override fun getLocation(location: JavaFileManager.Location): MutableIterable<File> =
        delegate.getLocation(location)

}